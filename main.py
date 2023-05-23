from App.UserInterface import App
import warnings


def main():
    ''' Main script which creates the App class interface and runs it. '''
    warnings.filterwarnings("ignore")
    app = App(usermode="dev")
    app.mainloop()


if __name__ == '__main__':
    main()
