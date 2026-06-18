DATA_DIR = 1
import os
if 'google.colab' in str(get_ipython()):
    # Running in Colab
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = "/content/drive/MyDrive/Teknisk fysik/Utbyte/Kurser/DAML/Project/notebooks"
else:
    # Running locally (Mac/Linux)
    DATA_DIR = '/Users/erikdalgard/Library/CloudStorage/GoogleDrive-dalgard.erik@gmail.com/My Drive/Teknisk fysik/Utbyte/Kurser/DAML/Project/notebooks'

os.chdir(DATA_DIR)
