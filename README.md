

Final Project – ML course 2026/27
Dataset description
The provided dataset consists of data from third year undergraduate students at the
Medical School at the University of Navarra. Several features, including basic data such
as age and sex, as well as metrics known to affect the development of Myopia are
provided for each student.
With the provided data the goal is to make four types of predictions:
- Binary classification to predict whether a student has myopia (M) or not. This is
encoded in column M, with labels YES and NO.
- Binary classification to predict whether a student has myopia magna (MM) or
not. This is encoded in column MM, with labels YES and NO. (MM es más de 6
dioptrias)
- Multi-Label classification, to combine both previous columns into one, denoted
as ‘COMBO’, where a student is classified as M (myopia), MM (myopia magna),
or C (control or no myopia).
- Multi-Label  classification, specified  in  column  ‘DCOMBO’,  where  myopia
students are further classified as type M1 or M2.
A training dataset is provided to train the models, as well as a testing dataset WITHOUT
LABELS. The ultimate goal is to produce labels in the test dataset that are as accurate as
possible, but all the process to achieve that goal will count towards the grade. You can
also use  libraries/functions/architectures  not  seen  in  class,  although  you  should
understand them. This will actually count positively! Just imagine you are in a company
and are given this dataset, and at the end of the project you have to present your work:
will you just show the accuracy on the validation set?
Group: In pairs (one group of 1 or 3 people if there is an odd number)
Deliverables: A zip folder containing at least the following files:
- The UNIQUE inferred labels for the test dataset stored in columns in a .TXT
format. Even if more than one model is presented, the final test labels should
correspond only to the model that you believe is performing the best (you can
use different models for the different classification tasks).
- A Jupyter notebook (or Python scripts) serving as a report, i.e., the notebook
should be commented (in English!). The notebooks should run with no errors.
Nevertheless, you should submit it already executed so that the output cells are
visible.
- All files needed for the program to run correctly without errors (except for the
training and test data which there are already available).


- The slides that you will use for the presentation. The slides should be prepared
as if the work was going to be presented in a conference (company), where you
briefly introduce the problem, how you tackled it, the methodology employed,
the results you obtained, and the conclusions.

- At the end of the presentation you should include some slides where you
specify: the work led by each of the members of the team (or which parts were
done together); if the work load split was 50/50 or different; if you used
Kears/PyTorch or both; and how you used generative AI models throughout the
work (e.g., to generate code, to get ideas about how to proceed, etc.).

- A 10’ video with the recorded presentation.
Submission: Via ADI
Deadline: Friday November 6 before 11.59pm.
Evaluation: The following aspects will be evaluated in the project:
## 1. Presentation.
- Accuracy achieved in the test.
- Methodology employed.
- A well-written report (within the Jupyter notebook) that justifies all actions and
functions.


