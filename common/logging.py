class Colors:
    HEADER = "\033[95m"
    INFO = "\033[94m"
    SUCCESS = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def header(message: str) -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {message}{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.RESET}")


def stage(name: str, message: str) -> None:
    print(f"{Colors.BOLD}[{name.upper()}]{Colors.RESET} {message}")


def info(message: str) -> None:
    print(f"{Colors.INFO}[INFO]{Colors.RESET} {message}")


def success(message: str) -> None:
    print(f"{Colors.SUCCESS}[OK]{Colors.RESET} {message}")


def warning(message: str) -> None:
    print(f"{Colors.WARNING}[WARN]{Colors.RESET} {message}")


def error(message: str) -> None:
    print(f"{Colors.ERROR}[ERROR]{Colors.RESET} {message}")
