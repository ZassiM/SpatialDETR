from App.UserInterface import App
from App.VideoGen import VideoGen
import warnings


def main():
    ''' Main script which creates the App class interface and runs it. '''
    warnings.filterwarnings("ignore")
    #app = App()
    app = VideoGen()
    app.mainloop()


if __name__ == '__main__':
    main()
