from modules.UI import UI
import warnings


def main():
    ''' Main script which creates the App class interface and runs it. '''
    warnings.filterwarnings("ignore")
    model = UI()
    model.load_from_config()
    debug = 0


if __name__ == '__main__':
    main()
