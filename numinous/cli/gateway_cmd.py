import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from numinous.cli.gateway_lib import config, manager

console = Console()


@click.group()
def gateway():
    """Manage the local API gateway

    \b
    Examples:
      crunch-numinous gateway start      # Start the gateway
      crunch-numinous gateway stop       # Stop the gateway
      crunch-numinous gateway status     # Check gateway status
      crunch-numinous gateway logs       # View gateway logs
      crunch-numinous gateway configure  # Set up API keys
    """
    pass


@gateway.command()
@click.option("--debug", is_flag=True, help="Log request/response details")
def start(debug):
    """Start the gateway (with API key setup if needed)"""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Starting Numinous Gateway[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    if manager.check_gateway_health():
        pid = manager.get_gateway_pid()
        console.print("[yellow]Gateway is already running[/yellow]")
        console.print(f"  [dim]PID:[/dim] {pid}")
        console.print(f"  [dim]URL:[/dim] {manager.GATEWAY_URL}")
        console.print()
        return

    env_vars = config.check_env_vars()
    all_env_ok = all(env_vars.values())

    set_keys = [key for key, ok in env_vars.items() if ok]
    missing_keys = [key for key, ok in env_vars.items() if not ok]

    if set_keys:
        console.print(f"[green]API keys set: {', '.join(set_keys)}[/green]")
    if missing_keys:
        console.print(f"[yellow]Missing API keys: {', '.join(missing_keys)}[/yellow]")
    console.print()

    success, pid, log_file = manager.start_gateway(debug=debug)
    if success:
        console.print()
        console.print(
            Panel.fit(
                f"[green]Gateway started![/green]\n\n"
                f"[dim]PID:[/dim] {pid}\n"
                f"[dim]URL:[/dim] {manager.GATEWAY_URL}\n"
                f"[dim]Logs:[/dim] {log_file.absolute()}\n\n"
                f"[yellow]View logs:[/yellow] [cyan]crunch-numinous gateway logs[/cyan]\n"
                f"[yellow]Stop:[/yellow] [cyan]crunch-numinous gateway stop[/cyan]",
                border_style="green",
                title="Gateway Running",
            )
        )
        console.print()
    else:
        console.print()
        console.print(
            Panel.fit(
                "[red]Failed to start gateway[/red]\n\n"
                "[yellow]Try checking the logs:[/yellow]\n"
                "   [cyan]crunch-numinous gateway logs[/cyan]",
                border_style="red",
            )
        )
        console.print()


@gateway.command()
def stop():
    """Stop the running gateway"""
    console.print()

    pid = manager.get_gateway_pid()
    if not pid:
        console.print("[yellow]Gateway is not running[/yellow]")
        console.print()
        return

    console.print(f"[cyan]Stopping gateway (PID: {pid})...[/cyan]")

    if manager.stop_gateway():
        console.print("[green]Gateway stopped[/green]")
    else:
        console.print("[red]Failed to stop gateway[/red]")
        console.print(f"[dim]Try manually: kill {pid}[/dim]")

    console.print()


@gateway.command()
@click.option("--debug", is_flag=True, help="Log request/response details")
@click.pass_context
def restart(ctx, debug):
    """Restart the gateway (stop + start)"""
    ctx.invoke(stop)
    ctx.invoke(start, debug=debug)


@gateway.command()
def status():
    """Show gateway status"""
    manager.show_gateway_status()


@gateway.command()
@click.option(
    "--no-follow",
    is_flag=True,
    help="Don't follow logs, just show last 50 lines",
)
def logs(no_follow):
    """View gateway logs (follows by default, press Ctrl+C to stop)"""
    manager.tail_logs(follow=not no_follow)


@gateway.command()
def configure():
    """Configure API keys"""
    console.print()
    console.print("[cyan]API Key Configuration[/cyan]")
    console.print()

    env_vars = config.check_env_vars()

    console.print("[dim]Current status:[/dim]")
    for key, is_set in env_vars.items():
        status = "[green]Set[/green]" if is_set else "[red]Not set[/red]"
        console.print(f"  {key}: {status}")
    console.print()

    all_set = all(env_vars.values())

    if all_set:
        if not Confirm.ask(
            "[bold cyan]All keys are configured. Update them?[/bold cyan]",
            default=False,
        ):
            console.print("[dim]No changes made[/dim]")
            console.print()
            return
        force_all = True
    else:
        force_all = False
        if any(env_vars.values()):
            if Confirm.ask(
                "[bold cyan]Update all keys (including existing ones)?[/bold cyan]",
                default=False,
            ):
                force_all = True

    if config.setup_api_keys(force_all=force_all):
        console.print("[green]API keys configured![/green]")
        console.print()

        if manager.check_gateway_health():
            if Confirm.ask(
                "[cyan]Gateway is running. Restart to load new keys?[/cyan]",
                default=True,
            ):
                console.print()
                if manager.stop_gateway():
                    console.print("[green]Stopped existing gateway[/green]")

                success, pid, log_file = manager.start_gateway()
                if success:
                    console.print()
                    console.print("[green]Gateway restarted with new keys![/green]")
                    console.print()
                else:
                    console.print("[red]Failed to restart gateway[/red]")
                    console.print()