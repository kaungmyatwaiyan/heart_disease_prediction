- **Complete the technical task in advance of your interview**, using the dataset provided.
- **Prepare a 15-minute presentation** summarising your approach, findings, and model.
- **Present your work during the interview**, followed by approximately 10 minutes of technical questions.

It is important that you note the following:

- In undertaking the test, you may use any tools you wish - including AI chatbots / coding assistants.
- The assessment will solely be based on what you present during the exercise - there is no need to provide additional material to the panel such as code or additional written documentation.
- The exercise should be your own work<sup>[\[1\]](#footnote-1)</sup>, please do not confer with anyone else. This may lead to elimination from the process.
- The response should be presented within 15 minutes.
- Following your presentation there will be approximately 10 minutes of questions from the panel. This will include discussion on the results you have presented, modelling and design choices you have made, and how the work might be further developed.
- The technical test will be followed by a competency interview, for which you may draw on evidence from other projects.
- The technical test will be conducted using Microsoft Teams and you will be asked to use the screen sharing functionality.

At interview, you will be assessed on the following:

- Identifying the key issues and patterns within the data.
- Your ability to tackle the problem using appropriate analytical techniques.
- Your coding approaches and appropriate use of tools<sup>[\[2\]](#footnote-2)</sup>.
- Your ability to convey the key information to a technical audience and to address the requirements set out in the exercise.
- Your ability to respond to further technical questions relating to the data / information presented.

Good luck and if you have any questions relating to the "Data Science Test Presentation" stage of the interview process, please get in touch with your line manager in the first instance.

# The Task

**Please read the instructions carefully prior to preparing your response.**

## The Data

In order to complete the test, you have been supplied with one csv file (you do not need any other data to complete the test):

- heart_disease_dataset.csv

The file contains the following fields:

| **Field** | **Description**                                                                |
| --------- | ------------------------------------------------------------------------------ |
| id        | unique ID for each patient                                                     |
| age       | age of the patient in years                                                    |
| sex       | Male/Female                                                                    |
| dataset   | place of study (data is drawn from four studies)                               |
| cp        | chest pain type (typical angina, atypical angina, non-anginal, asymptomatic)   |
| trestbps  | resting blood pressure in mm Hg on admission to the hospital                   |
| chol      | serum cholesterol in mg/dl                                                     |
| fbs       | fasting blood sugar > 120 mg/dl (True / False)                                 |
| restecg   | resting electrocardiographic results (normal, stt abnormality, lv hypertrophy) |
| thalch    | maximum heart rate achieved                                                    |
| exang     | exercise-induced angina (True / False)                                         |
| oldpeak   | ST depression induced by exercise relative to rest                             |
| slope     | the slope of the peak exercise ST segment (upsloping, flat, downsloping)       |
| ca        | number of major vessels (0-3) coloured by fluoroscopy                          |
| thal      | The thallium heart scan attribute refers to the results of a thallium heart scan, which is a type of nuclear imaging test used to evaluate blood flow to the heart muscle.  |
| **num**   | Heart disease indicator (0 to 4 with 0 indicating absence)                     |

## The Exercise

### Background

You are a data scientist working on a predictive model for patients presenting with chest pain, to identify potential heart disease patients.

You have access to the data detailed previously, drawn from several research studies of hospital patients.

Your task is to develop an initial predictive model using this dataset, and to assess its predictive power to identify cases of heart disease.

Note that you will need to create a binary target variable for your model using the 'num' field. An indicator score of zero represents absence of heart disease, any positive score indicates its presence.

You may use any tool or language to develop your model, but your model must be reproducible - which is to say it must be executable 'end-to-end', taking the source CSV file provided and then generating outputs.

You may want to use local development tools (for example R, Python, Jupyter notebooks, Excel) or a hosted service (for example, Google Colab <https://colab.research.google.com>). Use whatever tool you are comfortable with.

You may use AI tools including coding assistants if you wish. But bear in mind that the panel will expect you to be able to explain all aspects of how your solution operates and to justify the analytical and technical approaches you adopt.

Your task is to develop a single prototype model only. Prioritise demonstrating a working end-to-end solution and addressing the questions below over seeking to tune your model to provide the highest accuracy prediction.

**Presentation (15 minutes)**

You will be presenting your work back to a group of your peer data scientists to discuss your first phase of work and outline your proposed next steps. You may assume a good level of understanding of modelling and coding techniques, but the panel will not be familiar with the dataset you are working with.

Your presentation should cover:

- A brief introduction explaining the problem you are trying to solve, and how you structured your work.
- What data exploration did you conduct? What were your findings and how did this inform your approach?
- Why have you selected the modelling technique you use to generate predictions?
- A walk through of your code explaining the choices you have made. The panel will expect you to show the code during the presentation, but there is no requirement to execute the model 'live'.
- A discussion of the model performance - how have you measured this?
- Recommendations for how the work should be taken forward to refine your prototype model.

You will have a maximum of 15 minutes.

You may use whatever combination of visual aids that will help you communicate your work most effectively.

- The data for this exercise is well known as an open research dataset. Your preparation for this assessment may include researching work published by others (which you may find to be of mixed quality!). However, all code and analysis presented should have been conducted by you. You should mention where you have taken inspiration from previous work. [↑](#footnote-ref-1)

- Think about code quality in the sense of robustness, reusability, explainability & transparency. We do not expect code to be highly optimised from a computational efficiency perspective for this exercise. [↑](#footnote-ref-2)