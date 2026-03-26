import click


@click.group()
@click.version_option(package_name="crunch-numinous")
def main():
    """crunch-numinous — Numinous binary event forecasting toolkit"""
    pass


from numinous.cli.gateway_cmd import gateway  # noqa: E402

main.add_command(gateway)