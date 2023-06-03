from App.App import App
import warnings


def main():
    ''' Main script which creates the App class interface and runs it. '''
    warnings.filterwarnings("ignore")
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
