import json
import sys

import click

import spin
from spin import util
from spin.cmds.meson import (
    _get_configured_command,
    _set_pythonpath,
    build_dir_option,
    build_option,
)
from spin.cmds.util import run as _run


@click.command()
@click.option("-f", "--flag")
@click.option("-t", "--test", default="not set")
def example(flag, test, default_kwd=None):
    """🧪 Example custom command.

    Accepts arbitrary flags, and shows how to access `pyproject.toml`
    config.
    """
    click.secho("Running example custom command", bold=True, fg="bright_blue")
    print()
    config = util.get_config()
    commands = util.get_commands()
    click.secho("Flag provided with --flag is: ", fg="yellow", nl=False)
    print(flag or None)

    click.secho("Flag provided with --test is: ", fg="yellow", nl=False)
    print(test or None)

    click.secho(f"Default kwd is: {default_kwd}")

    click.secho("\nDefined commands:", fg="yellow")
    for section in commands:
        print(f"  {section}: ", end="")
        print(", ".join(cmd.name for cmd in commands[section]))

    click.secho("\nTool config is:", fg="yellow")
    print(json.dumps(config["tool.spin"], indent=2))


@click.command()
@click.argument("bench_args", nargs=-1)
@click.option("-s", "--no-error-supression", help="Extra test flag", is_flag=True)
@build_option
@build_dir_option
@click.pass_context
def bench(ctx, *, build=None, no_error_supression=None, build_dir=None, bench_args=None):
    """Run benchmarks in the build-install environment

    Builds the project and runs modab_root_finder/bench.py.
    Use --no-build to skip the build step.
    """
    if build:
        build_cmd = _get_configured_command("build")
        if build_cmd:
            click.secho(
                "Invoking `build` prior to running benchmarks:",
                bold=True,
                fg="bright_green",
            )
            ctx.invoke(build_cmd, build_dir=build_dir)

    args = []
    if no_error_supression:
        args += ['--no-error-supression']
    args += bench_args

    _set_pythonpath(build_dir)
    _run([sys.executable, "-P", "modab_root_finder/bench.py", *args])


@click.option("-e", "--extra", help="Extra test flag", type=int)
@util.extend_command(spin.cmds.meson.build, remove_args=("gcov",))
def build_ext(*, parent_callback, extra=None, **kwargs):
    """
    This version of build also provides the EXTRA flag, that can be used
    to specify an extra integer argument.
    """
    print(f"Preparing for build with {extra=}")
    parent_callback(**kwargs)
    print("Finalizing build...")
