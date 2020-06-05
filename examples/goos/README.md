# SPINS-B Examples
For examples mirroring code structure for using SPINS-B for your own designs, look at the python files in each of the folders. For interactive examples to familiarize yourself with the components of SPINS-B, look at the ipython-notebooks in the folder `interactive_colab`. The rest of this file explains how to setup these interactive colab files.

## Setting up Interactive Examples

### What is a colab notebook?
Colab is short for Colaboratory and it is google's platform for writing and executing python code in a notebook format in your web browser. We have chosen this platform for these interactive examples because you can install all the spins-b dependencies and access files all through your Google Drive (so, it becomes your operating system, essentially) and you don't have to worry about any local installation to run the code. More information is available at `colab.research.google.com`.

### How do I run the spins-b colab notebooks?

1. Download SPINS using the 'Download' ZIP on spins-b github at https://github.com/stanfordnqp/spins-b. Unzip the zip file to access the spins-b folder. 
Alternatively, you can also download SPINS by git clone:
$ git clone http://github.com/stanfordnqp/spins-b

2. Upload this spins-b folder you have just downloaded onto your google drive. You can choose what folder on your google drive you upload to (although it is easier if you just upload it to your `My Drive` folder if you can). 
3. Open a colab notebook by navigating in your google drive to  `spins-b/examples/interactive_colab`, double clicking on either of the `.ipynb` notebooks here, and clicking `Open with Google Colaboratory`. 
4. If you have uploaded `spins-b` in a subfolder of your `My Drive` folder, copy the path to this folder to the `folder_name` variable in the first cell of the colab notebook. 
5. Run the second cell of the colab notebook to install spins-b and all its dependencies.
6. Run the rest of the cells - everything else should run at this point.
