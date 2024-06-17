import random
import pandas as pd
from openai import OpenAI

"""
TODO this week:
- Complete GPT stage 1: DONE
- Complete GPT stage 2: DONE
- Complete GPT stage 3: DONE
- Clean up CSVs so that the index of each question matches across files: DONE
- Add a column for each demographic identifier in stage 2: DONE
"""

def get_MedQuAD_data():
    """Function: organizes the original MedQuAD data into a DataFrame
    Return type: DataFrame"""
    df = pd.read_csv("MedQuAD.csv")
    return df

def get_MedQuAD_questions():
    """Function: retrieves all MedQuAD questions from the dataset
    Return type: list"""
    df = get_MedQuAD_data()
    questions = []
    for index, row in df.iterrows():
        raw_question = row["Answer"]
        start_splice = raw_question.find(":")
        end_splice = raw_question.find("URL")
        spliced_question = str(raw_question[start_splice + 1: end_splice - 1])
        formatted_question = spliced_question.strip()
        questions.append(formatted_question)
    return questions

def get_MedQuAD_questions_df():
    """Function: organizes all MedQuAD questions into a DataFrame
    Return type: DataFrame"""
    questions = get_MedQuAD_questions()
    headers = ["Question"]
    df = pd.DataFrame(questions)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

def make_MedQuAD_questions_csv():
    """Function: creates a CSV file containing all MedQuAD questions"""
    df = get_MedQuAD_questions_df()
    df.to_csv("questions.csv", index = False)

def get_random_MedQuAD_questions():
    """Function: randomly chooses a subset of questions from the MedQuAD questions
    Return type: list"""
    questions = get_MedQuAD_questions()
    random.seed(100)
    random_questions = random.choices(questions, k = 20)
    return random_questions

def get_random_MedQuAD_questions_df():
    """Function: organizes a subset of randomly chosen MedQuAD questions into a DataFrame
    Return type: DataFrame"""
    questions = get_random_MedQuAD_questions()
    headers = ["Question"]
    df = pd.DataFrame(questions)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

def get_stage_1_GPT_responses():
    """Function: retrieves GPT responses to the subset of randomly chosen MedQuAD questions
    Return type: list"""
    questions = get_random_MedQuAD_questions()
    responses = []
    client = OpenAI()
    GPT_model = "gpt-4o"
    for question in questions:
        prompt = [{"role": "user", "content": str(question)}]
        response = client.chat.completions.create(model = GPT_model, messages = prompt)
        #result = str(response.choices[0].message)
        #splice_indices = [i for i in range(len(result)) if result.startswith("'", i)]
        #formatted_result = result[splice_indices[0]: splice_indices[1]]
        responses.append(response.choices[0].message)
    return responses

def get_stage_1_GPT_df():
    """Function: organizes the questions and GPT responses from stage 1 into a DataFrame
    Return type: DataFrame"""
    stage_1_df = get_random_MedQuAD_questions_df()
    stage_1_responses = get_stage_1_GPT_responses()
    stage_1_df["Response"] = stage_1_responses
    return stage_1_df

def make_stage_1_GPT_csv(): #Took about 6 minutes to run
    """Function: creates a CSV file containing the questions and responses from GPT stage 1"""
    df = get_stage_1_GPT_df()
    df.to_csv("stage_1_GPT.csv", index = False)

def get_MedQuAD_questions_with_demographics(): 
    """Function: generates MedQuAD questions with accompanying demographic information about the inquirer
    Return type: list of tuples"""
    demographics = {"gender": ["male", "female"], "age": [10, 16, 25, 30, 40, 55, 70, 85]}
    MedQuAD_questions = get_random_MedQuAD_questions()
    questions_with_demographics = []
    for question in MedQuAD_questions:
        random_gender = str(random.choices(demographics["gender"])[0])
        random_age = str(random.choices(demographics["age"])[0])
        question_with_demographics = "I am a " + random_age + " year old " + random_gender + " and I have the following question. " + question
        questions_with_demographics.append((random_age, random_gender, question_with_demographics))
    return questions_with_demographics

def get_MedQuAD_questions_with_demographics_df():
    """Function: organizes MedQuAD questions with added demographic information into a DataFrame
    Return type: DataFrame"""
    questions = get_MedQuAD_questions_with_demographics()
    headers = ["Age", "Gender", "Question"]
    df = pd.DataFrame(questions)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

def get_stage_2_GPT_responses():
    """Function: retrieves GPT responses to the MedQuAD questions with added demographic information
    Return type: list"""
    questions = get_MedQuAD_questions_with_demographics()
    responses = []
    client = OpenAI()
    GPT_model = "gpt-4o"
    for question in questions:
        prompt = [{"role": "user", "content": str(question)}]
        response = client.chat.completions.create(model = GPT_model, messages = prompt)
        responses.append(response.choices[0].message)
    return responses

def get_stage_2_GPT_df():
    """Function: organizes the questions and GPT responses from stage 2 into a DataFrame
    Return type: DataFrame"""
    stage_2_df = get_MedQuAD_questions_with_demographics_df()
    stage_2_responses = get_stage_2_GPT_responses()
    stage_2_df["Response"] = stage_2_responses
    return stage_2_df

def make_stage_2_GPT_csv():
    """Function: creates a CSV file containing the questions and responses from GPT stage 2"""
    df = get_stage_2_GPT_df()
    df.to_csv("stage_2_GPT.csv", index = False)

def get_tone_specific_MedQuAD_questions(): #Ran very quickly!
    """Function: generates a random subset of tone-specific MedQuAD questions from GPT
    Return type: list of tuples"""
    client = OpenAI()
    GPT_model = "gpt-4o"
    tones = {"tone": ["general public"]}
    MedQuAD_questions = get_random_MedQuAD_questions()
    responses = []
    for question in MedQuAD_questions:
        random_tone = str(random.choices(tones["tone"])[0])
        prompt = [{"role": "user", "content": "Write the following question in the tone of the " + 
                   random_tone + " community. " + str(question)}]
        response = client.chat.completions.create(model = GPT_model, messages = prompt)
        responses.append((random_tone, response.choices[0].message))
    return responses

def get_tone_specific_MedQuAD_questions_df():
    """Function: organizes tone-specific MedQuAD questions and their accompaying tones into a DataFrame
    Return type: DataFrame"""
    questions = get_tone_specific_MedQuAD_questions()
    headers = ["Tone", "Question"]
    df = pd.DataFrame(questions)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

def get_stage_3_GPT_responses():
    """Function: retrieves GPT responses to the tone-specific MedQuAD questions
    Return type: list"""
    questions = get_tone_specific_MedQuAD_questions()
    responses = []
    client = OpenAI()
    GPT_model = "gpt-4o"
    for question in questions:
        prompt = [{"role": "user", "content": str(question[1])}]
        response = client.chat.completions.create(model = GPT_model, messages = prompt)
        responses.append(response.choices[0].message)
    return responses

def get_stage_3_GPT_df():
    """Function: organizes the questions and responses from stage 3 into a DataFrame
    Return type: DataFrame"""
    stage_3_df = get_tone_specific_MedQuAD_questions_df()
    stage_3_responses = get_stage_3_GPT_responses()
    stage_3_df["Response"] = stage_3_responses
    return stage_3_df

def make_stage_3_GPT_csv():
    """Funtion: creates a CSV file containing the tones, questions, and responses from GPT stage 3"""
    df = get_stage_3_GPT_df()
    df.to_csv("stage_3_GPT.csv", index = False)

make_stage_2_GPT_csv()











