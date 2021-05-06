# Stock Market Analysis & prediction

## Course IT314 - Group project

## Project Description:

This is a repository for the course project in the course IT314 Software Engineering offered in Winter 2021, at DA-IICT. 
Our aim is to create a web application that analyzes previous stock data of certain companies, with help of certain parameters that affects the stock values. 
And after that our application predicts the future value of the selected stock. This new methodology of analyzing the stock values will be of great help for 
new budding investors and as well as for professionals for doing data analysis quickly. 



## Installing dependencies

**`pip install flask`**

**`pip install numpy`**

**`pip install matplotlib`**

**`pip install pandas`**

**`pip install datetime`**

**`pip install keras`**

**`pip install tensorflow`**

**`pip install sklearn`**

**`pip install time`**

**`pip install csv`**

**`pip install math`**

**`pip install datetime`**

**`pip install dateutil.relativedelta`**



## Pull requests or contribution guidelines

Follow this guide step by step to make a contributon to the project. Don't directly merge your code in the main branch or merge a random pull request.

1. Fork this main repository in your github profile. This will create a copy of this repo in your profile. You'll edit your code there. Click on the fork button on the top of this repository. (Only need to do this step once)

2. Clone the forked repo on your profile to your local laptop as local repo. (Only need to do this step once)

**`git clone <url of your forked repo>`**

And then move into your repo by using GitBash.

**`cd 11-tagged-news`**

3. Set this repo as the remote upstream. This will help further to sync your local repo with the updated version. (Only need to do this step once) Learn more

**`git remote add upstream https://github.com/Dev4522v/Group12-StockMarketAnalysis.git`**

4. Before adding your changes, create a new feature branch so that the code in the main is not changed. Follow the given convention for the branch name. (Only need to do this step once if a branch is not created)

**`git checkout -b dev-<YOUR NAME>`**

5. If there has been any changes in the main repo after you have cloned your repo, make sure to sync your repo using this guide. Or you can simply run this command, this will update your main from main repo, so make sure your feature branch also has this changes and then start working.

**`git pull upstream main`**

6. Add, commit and push your changes in dev-branch in your local repository. (Do this step after making your changes)

**`git add -A`**

**`git commit -m "<COMMIT MESSAGE>"`**

7. This step will update your forked repo with the changes that you made in your dev-branch in your local repo. Learn more about pushing commits to remote repo

**`git push -u origin <BRANCH NAME>`**

8. Go to your forked repo in your profile and make a pull request once you are done with the above steps. Follow this link to know more about this step. NOTE: While doing a pull request, keep the base branch as dev- in the main repo as well. This branch name will be same as your the branch name that you worked on your local and forked repo. If possible tag the issue number that you are dealing with in the description.
