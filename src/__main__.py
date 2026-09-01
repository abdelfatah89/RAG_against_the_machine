from fire import Fire

from .cli import CLI


def main() -> None:
    cli = CLI()
    Fire(cli)


if __name__ == "__main__":
    main()
