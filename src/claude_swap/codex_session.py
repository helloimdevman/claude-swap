"""Session mode for Codex: run Codex as a stored account in one terminal.

``cswap codex run NUM|EMAIL`` launches ``codex`` with ``CODEX_HOME`` pointing
at a persistent per-account profile under
``<backup_dir>/sessions/<num>-<email-slug>/`` (``<backup_dir>`` is
``<root>/codex``), leaving the default login and every other Codex process
untouched. Everything not Codex-specific is ``session.SessionManager``'s: the
pre-launch refresh through the consume gate, the stale marker, the bootstrap
lock, the share manifest and history merge.

What differs from Claude:

- The launch is ``codex -c cli_auth_credentials_store="file" <args>``. The
  forced file store keeps the profile's login in ``<profile>/auth.json`` (the
  Keychain item a keyring config would pick is keyed by the home's path, and
  cswap never writes Codex's Keychain), and any ``-c`` makes the TUI run an
  in-process app-server instead of the shared per-home daemon
  (tui/src/daemon_startup.rs:54-89), so the session's auth lives in the
  process cswap launched.
- Validity is offline: the profile's auth.json parses, carries
  ``last_refresh`` and ``tokens.account_id`` (Codex sends no bearer, and
  refuses to refresh, without them) and names the slot's account. There is
  no ``claude auth status`` probe, so no "unknown"/"unreachable" verdicts.
- Codex keeps no per-process session records, so liveness comes from three
  places (any one is enough, unreadable counts as live): the record
  ``cswap codex run`` writes before it execs (the pid survives the exec), the
  profile's app-server daemon pid file, and a ``ps`` scan for a ``codex``
  whose argv names the profile.
- ``--share-history`` also points ``CODEX_SQLITE_HOME`` at the shared home:
  the ``codex resume`` picker lists threads from the SQLite ``threads`` table
  first and only scans ``sessions/`` when that is empty (codex-spec §12b).
- No MCP mirror: Codex keeps MCP servers in the shared ``config.toml``. In a
  Codex profile ``.credentials.json`` is Codex's MCP OAuth store, not the
  account login, and is never touched.

Must not import ``switcher`` or ``codex_switcher`` (they import us).
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import NoReturn

from claude_swap import codex_auth, codex_store, process_detection
from claude_swap.codex_store import CodexStoreUnsupported
from claude_swap.exceptions import SessionError
from claude_swap.printer import warning
from claude_swap.process_detection import CodexProcess
from claude_swap.session import SessionManager
from claude_swap.settings import atomic_write_json

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

# Shared from the Codex home (codex-spec §9): user configuration only. Not
# auth.json (the account), not plugins/ (holds a per-install app-server).
SHARED_ITEMS = (
    "config.toml",
    "AGENTS.md",
    "AGENTS.override.md",
    "skills",
    "rules",
    "hooks.json",
    "agents",
    "themes",
)

# Linked additionally under --share-history: rollouts, thread names and
# prompt history. The threads DB follows via CODEX_SQLITE_HOME.
HISTORY_ITEMS = (
    "sessions",
    "archived_sessions",
    "session_index.jsonl",
    "history.jsonl",
)

# Env vars that would replace the profile's login inside codex
# (login/src/auth/manager.rs:1487-1596): scrubbed from the session launch.
AUTH_OVERRIDE_ENV_VARS = ("CODEX_API_KEY", "CODEX_ACCESS_TOKEN")

# The default home a session was launched from ("" = the implicit
# ~/.codex), set in the session's environment so a nested run can restore it:
# the launch replaced the user's own CODEX_HOME with the profile.
DEFAULT_HOME_ENV = "CSWAP_CODEX_DEFAULT_HOME"

# One `{pid, procStart}` record per `cswap codex run` that exec'd into the
# profile — per pid, so two runs joining one profile never overwrite each
# other's record (a lost record would let a later bootstrap rewrite auth.json
# under the first session).
RUN_RECORDS = ".cswap-run"

# The profile's app-server daemon pid file, current and legacy names
# (app-server-daemon/src/lib.rs:50-56).
DAEMON_PID_FILES = ("daemon.pid", "app-server.pid")

# A record that cannot be read: JSON/Unicode errors are ValueError; a
# non-object, a missing or mistyped pid raise the rest.
_RECORD_ERRORS = (
    OSError, ValueError, KeyError, TypeError, AttributeError,
    OverflowError, RecursionError,
)


def read_session_credentials(session_dir: Path) -> str | None:
    """The profile's ``auth.json`` text when it is a JSON object, else None.

    Codex rewrites auth.json in place, so a parse failure may be a write in
    flight: retried briefly, like the live store's read.
    """
    path = session_dir / "auth.json"
    for attempt in range(codex_store._PARSE_RETRY_ATTEMPTS):
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError):
            return None
        except UnicodeDecodeError:
            text = None
        except OSError:
            return None
        if codex_store._json_object(text) is not None:
            return text
        if attempt + 1 < codex_store._PARSE_RETRY_ATTEMPTS:
            time.sleep(codex_store._PARSE_RETRY_DELAY)
    return None


def read_session_identity(session_dir: Path) -> tuple[str, str] | None:
    """``(email, account_id)`` the profile is logged in as, from its
    id_token; None when unreadable (which drift checks read as "no drift")."""
    creds = read_session_credentials(session_dir)
    ident = codex_auth.identity(creds) if creds else None
    if not ident or not ident["email"]:
        return None
    return ident["email"], ident["organizationUuid"] or ""


def login_problem(creds: str, email: str, org_uuid: str) -> str | None:
    """Why Codex could not run ``creds`` as the slot ``(email, org_uuid)``,
    or None. Offline, from the blob alone."""
    data = codex_auth.parse_blob(creds)
    ident = codex_auth.identity(creds)
    if data is None or ident is None or not ident["email"]:
        return "not a ChatGPT login"
    if not data.get("last_refresh"):
        return "no last_refresh"
    if not (data.get("tokens") or {}).get("account_id"):
        return "no tokens.account_id"
    if ident["email"] != email or (org_uuid and ident["organizationUuid"] != org_uuid):
        return f"it is {ident['email']} ({ident['organizationUuid']})"
    return None


def _daemon_matches(pid: int, record: dict) -> bool:
    """Is the live ``pid`` still the daemon its pid file describes?

    Unknowable answers True (live). Codex's native ``processIdentity``
    (app-server-daemon/src/backend/pid_identity.rs) is locale- and
    timezone-free: Linux start ticks, compared exactly; macOS epoch start
    seconds, where only a process that started later disqualifies (as in
    ``pid_matches_record``).

    ponytail: a legacy record (no ``processIdentity``) carries only ``ps -o
    lstart`` text in the daemon's own locale and time zone, so a live pid
    counts as the daemon; a recycled pid keeps its profile "live" until that
    process exits.
    """
    ident = record.get("processIdentity")
    if not isinstance(ident, dict):
        return True
    if "startTicks" in ident:
        return process_detection.pid_matches_record(pid, str(ident["startTicks"]))
    if "startSeconds" in ident:
        started = process_detection.process_started_at(pid)
        return started is None or started <= int(ident["startSeconds"]) + (
            process_detection.PID_REUSE_SLACK_S
        )
    return True


def _reservation_is_stale(pid_file: Path) -> bool:
    """For an EMPTY daemon pid file: whether no daemon start holds its
    reservation, i.e. a start died and nothing runs.

    Mirrors Codex (app-server-daemon/src/backend/pid.rs:264-294, 631-718): a
    starting daemon holds ``flock(LOCK_EX)`` on ``<pid file>.lock`` until it
    writes the pid. Taking that lock non-blocking succeeds only when nobody
    holds it (closing the fd releases it at once). A held lock, or one that
    cannot be tested (Windows, an unopenable lock file), is not stale.

    Like Codex, the pid file is re-read under the lock: a daemon may have
    written its pid and released the lock since the caller read it empty,
    and that is a running daemon, not a dead start.
    """
    if fcntl is None:
        return False
    try:
        fd = os.open(pid_file.with_name(pid_file.name + ".lock"), os.O_RDONLY)
    except FileNotFoundError:
        return True  # Codex creates it before reserving: never reserved
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return not pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return True  # removed meanwhile: nothing reserved, nothing running
    except (OSError, ValueError):  # BlockingIOError: a daemon is starting
        return False
    finally:
        os.close(fd)


def scan_live_sessions(session_dir: Path) -> tuple[list[CodexProcess], int]:
    """Codex processes running against a profile, and records that could
    not be read (every caller gates a destructive step: not knowing is not
    "nothing there")."""
    if not session_dir.exists():
        return [], 0
    live: dict[int, CodexProcess] = {}
    unreadable = 0

    def check(path: Path, matches, daemon: bool = False) -> None:
        nonlocal unreadable
        try:
            text = path.read_text(encoding="utf-8")
            if daemon and not text.strip():
                # Codex reserves the pid file empty while a daemon starts. No
                # pid to report: a live (or undeterminable) reservation counts
                # with the unreadable records, a stale one counts as nothing.
                unreadable += not _reservation_is_stale(path)
                return
            data = json.loads(text)
            pid = data["pid"]
            if process_detection.is_pid_alive(pid) and matches(pid, data):
                live[pid] = CodexProcess(pid=pid, args="")
        except FileNotFoundError:
            pass  # pruned or never written: not evidence of anything
        except _RECORD_ERRORS:
            unreadable += 1

    for path in sorted((session_dir / RUN_RECORDS).glob("*.json")):
        check(path, lambda pid, d: process_detection.pid_matches_record(
            pid, d.get("procStart")
        ))
    for name in DAEMON_PID_FILES:
        check(session_dir / "app-server-daemon" / name, _daemon_matches, daemon=True)

    # A daemon installed under the profile runs from <profile>/packages/…
    # (Codex canonicalizes CODEX_HOME, hence the resolved spelling too).
    # ponytail: one `ps -A` per scan of an existing profile, and a usage pass
    # scans each profile a few times; cache the listing per pass if it shows.
    spellings = {str(session_dir), str(session_dir.resolve())}
    for proc in process_detection.list_codex_processes():
        if any(s in proc.args for s in spellings):
            live.setdefault(proc.pid, proc)
    return list(live.values()), unreadable


def profile_is_quiescent(session_dir: Path) -> bool:
    """Nothing runs against the profile AND every record could be read."""
    sessions, unreadable = scan_live_sessions(session_dir)
    return not sessions and unreadable == 0


def write_run_record(session_dir: Path) -> None:
    """Record this process — about to exec codex under the same pid — as
    live in the profile. Records of runs that have exited are pruned
    (readable and pid gone); unreadable ones stay, as the scan reads them."""
    records = session_dir / RUN_RECORDS
    for path in records.glob("*.json"):
        try:
            gone = not process_detection.is_pid_alive(
                json.loads(path.read_text(encoding="utf-8"))["pid"]
            )
        except _RECORD_ERRORS:
            continue
        if gone:
            # Pruning is housekeeping: a record we cannot delete must not
            # abort this launch.
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
    pid = os.getpid()
    atomic_write_json(
        records / f"{pid}.json",
        {"pid": pid, "procStart": process_detection.proc_start_stamp(pid)},
    )


class CodexSessionManager(SessionManager):
    """``SessionManager`` with Codex's launch, seed, validity and sharing."""

    shared_items = SHARED_ITEMS
    history_items = HISTORY_ITEMS
    binary = "codex"
    auth_override_env_vars = AUTH_OVERRIDE_ENV_VARS

    # -- launch ------------------------------------------------------------
    #
    # `SessionManager.run` drives the launch; these are its provider seams.

    def _normalize_nested_env(self) -> None:
        """Undo a session's CODEX_HOME when this runs inside one.

        A CODEX_HOME inside a profile means a session's environment (codex's
        own shell commands inherit it). It is replaced, for the rest of this
        process (which execs right after with an environment built from
        ``os.environ``), by the default home recorded when that session was
        launched (DEFAULT_HOME_ENV), or dropped when that default was the
        implicit ``~/.codex``. The live-login read, the consume gate's
        live-lineage guard, the fast path, sharing and the launch then all see
        the user's real default home — otherwise the gate would compare a slot
        against the profile's login (or a stale ``~/.codex``) and could spend
        the real default login's grant or seed a second copy of it.
        """
        preset = os.environ.get("CODEX_HOME")
        # Popped either way: a plain-codex fast path must not carry it, and
        # `_exec_session` records the current default for the next session.
        recorded = os.environ.pop(DEFAULT_HOME_ENV, "")
        if not preset or not self._in_profile(Path(preset)):
            return
        if recorded and not self._in_profile(Path(recorded)):
            os.environ["CODEX_HOME"] = recorded
        else:
            del os.environ["CODEX_HOME"]
        warning(
            f"CODEX_HOME points at a session profile ({preset}); using the "
            f"default Codex home ({codex_store.codex_home()}) for this launch."
        )

    def _fast_path_allowed(self) -> bool:
        """Always (after ``_normalize_nested_env``): an exported CODEX_HOME
        outside the session profiles IS the user's default login — cswap's
        live store reads it too — so the same-account fast path holds there.
        It matters more than for Claude: Codex rotates the refresh token on
        every refresh, and a second copy of the live login breaks whichever
        copy refreshes second."""
        return True

    def _current_login(self) -> tuple[str, str] | None:
        """The default login, or None when its store is one cswap cannot read
        (``ephemeral``, or keyring/auto off macOS). The session itself needs
        no live store — the launch forces the file store, and the consume gate
        already declines to refresh while the live login is unreadable — so
        it proceeds without the fast path, saying what it could not check."""
        try:
            return super()._current_login()
        except CodexStoreUnsupported as e:
            warning(
                f"Could not read the default Codex login ({e}). If this account "
                "is that login, its two copies will contend for one refresh "
                "token."
            )
            return None

    def _exec_session(
        self,
        codex_bin: str,
        session_dir: Path,
        codex_args: list[str],
        share_history: bool,
    ) -> NoReturn:
        """``codex -c cli_auth_credentials_store="file" <args>`` with
        ``CODEX_HOME=<profile>``, after recording the run."""
        drop = (*AUTH_OVERRIDE_ENV_VARS, "CODEX_SQLITE_HOME")
        env = {k: v for k, v in os.environ.items() if k not in drop}
        # Kept in the session's env on purpose (a path, nothing secret): it
        # is how a nested `cswap codex run` finds the default home again.
        env[DEFAULT_HOME_ENV] = os.environ.get("CODEX_HOME", "")
        env["CODEX_HOME"] = str(session_dir)
        if share_history:
            env["CODEX_SQLITE_HOME"] = str(self._share_source())
        try:
            write_run_record(session_dir)
        except OSError as e:
            # Unrecorded, the session would be invisible to every guard that
            # protects its auth.json (re-bootstrap, removal, purge).
            raise SessionError(
                f"Could not record the session in {session_dir}: {e}"
            ) from e
        self._exec(
            codex_bin, ["-c", codex_store.FILE_STORE_OVERRIDE, *codex_args], env=env
        )

    def _in_profile(self, path: Path) -> bool:
        return path.resolve().is_relative_to(self.sessions_dir.resolve())

    # -- bootstrap -----------------------------------------------------------

    @staticmethod
    def _has_refresh_token(creds: str) -> bool:
        return bool((codex_auth.oauth_view(creds) or {}).get("refreshToken"))

    def _bootstrap(
        self, session_dir: Path, account_num: str, email: str, org_uuid: str
    ) -> None:
        """Seed ``<profile>/auth.json`` (0600) from the slot backup. Caller
        holds the lock. No Keychain step: the launch forces the file store.

        The backup is checked first with the same offline test validity
        uses, so a login Codex could not run is refused here, before the
        base's failed-validation cleanup would delete the profile (and the
        history it holds).
        """
        creds, unreadable = self.switcher._read_account_credentials_ex(
            account_num, email
        )
        relogin = f"cswap codex add --login --slot {account_num}"
        if not creds:
            if unreadable:
                raise SessionError(
                    f"Account-{account_num}'s backup is in the macOS Keychain "
                    f"but it is unreadable right now (locked or no GUI "
                    f"session). Retry from a GUI terminal; do not re-add."
                )
            raise SessionError(
                f"Account-{account_num} has no stored credentials. "
                f"Re-add with: {relogin}"
            )
        problem = login_problem(creds, email, org_uuid)
        if problem:
            raise SessionError(
                f"Account-{account_num}'s stored login cannot start a session "
                f"({problem}). Re-add with: {relogin}"
            )
        # atomic_write_json: mkstemp 0600, chmod of the profile dir to 0700.
        atomic_write_json(session_dir / "auth.json", json.loads(creds))
        self._logger.info(
            f"Bootstrapped Codex session profile for account {account_num} "
            f"at {session_dir}"
        )

    # -- validation ------------------------------------------------------------

    def _session_validity(self, session_dir: Path, email: str, org_uuid: str) -> str:
        """``"valid"`` or ``"invalid"``, offline (see the module docstring)."""
        creds = read_session_credentials(session_dir)
        if creds is None or login_problem(creds, email, org_uuid):
            return "invalid"
        return "valid"

    # -- sharing ---------------------------------------------------------------

    def _share_source(self) -> Path:
        """The Codex home (a nested run has already dropped a CODEX_HOME that
        pointed into a profile; see ``_fast_path_allowed``)."""
        return codex_store.codex_home()

    def _sync_mcp_servers(self, session_dir: Path, share: bool) -> None:
        """No mirror: Codex's MCP servers live in the shared config.toml."""
