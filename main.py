from keep_alive import keep_alive
from ap_main import setup_bot, start_bot

def main():
    keep_alive()
    updater = setup_bot()
    start_bot(updater)

if __name__ == "__main__":
    main()
