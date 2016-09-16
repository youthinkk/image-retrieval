# CS2108 Assignment 1 - Image Retrieval
A web app to index, match and retrieve images based on a range of visual, concept and text features. In particular, techniques such as color histogram (CH), visual keyword (VW), visual concept (VC), deep learning (DL) and text (TEXT) features are used for retrieval.

# Development
### Dependencies
* Python 2.7
* OpenCV 2.4.11
* [PyCharm 2016.2](https://www.jetbrains.com/pycharm/)

### Setup
1. Open **PyCharm**
2. Choose **GitHub** under **Check out from Version Control**
3. Under **Clone Repository**, input the following details:
  * **Git Repository URL** : `https://github.com/youthinkk/image-retrieval.git`
  * **Directory Name** : `cs2108`
4. After the repository is cloned, choose **Edit Configurations** under **Run** at the tool bar
5. Enable **Run browser** and configure the path to `http://127.0.0.1:8000/retrieval`
6. Test the setting by running `cs2108` module
7. You are all set now :smile:

### Files
* [cs2108/](/cs2108) : Python package for the project
* [cs2108/settings.py](/cs2108/settings.py) : Settings/configuration for this Django project
* [cs2108/urls.py](/cs2108/urls.py) : The URL declarations for this Django project; a “table of contents” of the Django-powered site
* [cs2108/wsgi.py](/cs2108/wsgi.py) : An entry-point for WSGI-compatible web servers to serve the project
* [retrieval/](/retrieval) : Main backend code of the project
* [retrieval/data/](/retrieval/data) : Image data for training and testing
* [retrieval/src/](/retrieval/src) : Algorithms for indexing, matching and retrieving images
* [templates/](/templates) : HTML for the web pages

# Authors
* [Wu Yu Ting](https://github.com/youthinkk)
* [Jiang Hongchao](https://github.com/jianghc93)
* [Donald Shum](https://github.com/donaldshum)
