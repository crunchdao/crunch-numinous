import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from rich.console import Console

from numinous.cli.gateway_lib.config import load_env_file

console = Console()

GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8090"))
GATEWAY_URL = f"http://localhost:{GATEWAY_PORT}"

PID_FILE = Path.home() / ".crunch-numinous-gateway.pid"
LOG_FILE = Path.home() / ".crunch-numinous-gateway.log"


def check_gateway_health() -> bool:
    try:
        req = urllib.request.Request(f"{GATEWAY_URL}/api/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_gateway_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # check if alive
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return None


def stop_gateway() -> bool:
    pid = get_gateway_pid()
    if not pid:
        return False

    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                PID_FILE.unlink(missing_ok=True)
                return True
        os.kill(pid, signal.SIGKILL)
        PID_FILE.unlink(missing_ok=True)
        return True
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def start_gateway(debug: bool = False) -> tuple[bool, int | None, Path | None]:
    try:
        log_handle = open(LOG_FILE, "a")

        # Build environment: package defaults < saved user keys < live env vars
        from dotenv import dotenv_values
        import numinous.gateway

        defaults_env = Path(numinous.gateway.__file__).parent / "defaults.env"

        env = os.environ.copy()
        if debug:
            env["GATEWAY_DEBUG"] = "1"
        # 1. Load package defaults (lowest priority)
        for k, v in dotenv_values(defaults_env).items():
            if v:
                env.setdefault(k, v)
        # 2. Load user-saved keys (middle priority, won't override live env)
        for k, v in load_env_file().items():
            env.setdefault(k, v)

        api_keys_in_env = [k for k in env if k.endswith("_API_KEY") and env[k]]
        console.print(f"  [dim]Env keys passed to gateway: {', '.join(api_keys_in_env) or 'none'}[/dim]")

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "numinous.gateway.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(GATEWAY_PORT),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

        PID_FILE.write_text(str(process.pid))

        console.print("  [cyan]Starting gateway...[/cyan]", end="")
        for _ in range(15):
            time.sleep(0.5)
            if check_gateway_health():
                console.print(" [green]\u2713[/green]")
                return True, process.pid, LOG_FILE
            console.print(".", end="")

        console.print(" [red]\u2717[/red]")
        log_handle.close()
        return False, None, None

    except Exception as e:
        console.print(f" [red]\u2717[/red] Error: {e}")
        return False, None, None


def show_gateway_status() -> None:
    console.print()
    console.print("[cyan]Gateway Status[/cyan]")
    console.print()

    is_healthy = check_gateway_health()
    pid = get_gateway_pid()

    if is_healthy and pid:
        console.print("  [green]\u2713[/green] Running")
        console.print(f"  [dim]URL:[/dim] {GATEWAY_URL}")
        console.print(f"  [dim]PID:[/dim] {pid}")
        console.print(f"  [dim]Logs:[/dim] {LOG_FILE.absolute()}")
        console.print()
        console.print("  [yellow]View logs:[/yellow] [cyan]crunch-numinous gateway logs[/cyan]")
        console.print("  [yellow]Stop:[/yellow] [cyan]crunch-numinous gateway stop[/cyan]")
    elif pid:
        console.print("  [yellow]![/yellow] Process running but not responding")
        console.print(f"  [dim]PID:[/dim] {pid}")
        console.print()
        console.print("  [yellow]Stop:[/yellow] [cyan]crunch-numinous gateway stop[/cyan]")
    else:
        console.print("  [red]\u2717[/red] Not running")
        console.print()
        console.print("  [yellow]Start:[/yellow] [cyan]crunch-numinous gateway start[/cyan]")

    console.print()


def tail_logs(follow: bool = True) -> None:
    if not LOG_FILE.exists():
        console.print()
        console.print(f"[yellow]! Log file not found: {LOG_FILE}[/yellow]")
        console.print()
        return

    try:
        if follow:
            subprocess.run(["tail", "-f", str(LOG_FILE)])
        else:
            subprocess.run(["tail", "-n", "50", str(LOG_FILE)])
    except KeyboardInterrupt:
        console.print()
        console.print("[dim]Log viewing stopped[/dim]")
        console.print()
    except Exception as e:
        console.print()
        console.print(f"[red]\u2717 Error viewing logs: {e}[/red]")
        console.print()