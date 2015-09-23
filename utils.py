class ascii_colors:
    HEADER = '\033[95m'

    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GRAY = '\033[2m'

    ENDC = '\033[0m'

    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'


def format_message(content, color):
    return '%s%s%s' % (color, content, ascii_colors.ENDC)


def warning(content):
    return format_message(content, ascii_colors.YELLOW)


def error(content):
    return format_message(content, ascii_colors.RED)


def info(content):
    return format_message(content, ascii_colors.BLUE)


def debug(content):
    return format_message(content, ascii_colors.GRAY)


def announce(content):
    return format_message(ascii_colors.BOLD + content, ascii_colors.GREEN)
