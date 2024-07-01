import random
import pandas as pd
from openai import OpenAI

"""
TODO this week (and next Monday)
- Create simplified dataset with age and gender removed:
- Add function comments for new functions:
"""
def get_USMLE_data():
    """Function: organizes the original USMLE data into a DataFrame
    Return type: DataFrame"""
    all_data = {"train": "data/train-00000-of-00001-a6e6f83961340098.parquet", "test": "data/test-00000-of-00001-997553f5b24c9767.parquet"}
    df = pd.read_parquet("hf://datasets/augtoma/medqa_usmle/" + all_data["train"])
    return df

def get_USMLE_questions():
    """Function: retrieves all USMLE questions from the dataset
    Return type: list"""
    df = get_USMLE_data()
    questions = []
    for index, row in df.iterrows():
        question = row["question"]
        questions.append(question)
    return questions

def get_USMLE_questions_df():
    """Function: organizes all USMLE questions into a DataFrame
    Return type: DataFrame"""
    questions = get_USMLE_questions()
    headers = ["Question"]
    df = pd.DataFrame(questions)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

"""def get_simplified_USMLE_questions_df():
    questions = get_simplified_USMLE_questions()
    headers = ["Question"]
    df = pd.DataFrame(questions)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df"""

def make_USMLE_questions_csv():
    """Function: creates a CSV file containing all USMLE questions"""
    df = get_USMLE_questions_df()
    df.to_csv("questions.csv", index = False)

"""def make_simplified_USMLE_questions_csv():
    df = get_simplified_USMLE_questions_df()
    df.to_csv("simplified_questions.csv", index = False)"""

def get_random_USMLE_questions():
    """Function: randomly chooses a subset of questions from the USMLE questions
    Return type: list"""
    questions = get_USMLE_questions()
    random.seed(10)
    random_questions = random.choices(questions, k = 20)
    return random_questions

def get_random_simplified_USMLE_questions():
    original_questions = get_random_USMLE_questions()
    simplified_questions = []
    client = OpenAI()
    GPT_model = "gpt-4o"
    for question in original_questions:
        prompt = [{"role": "system", "content": "You are rewriting medical questions. Rewrite the question provided with age and gender removed. Change pronouns to be gender neutral."}, {"role": "user", "content": "Rewrite this question: " + question}]
        response = client.chat.completions.create(model = GPT_model, messages = prompt)
        simplified_questions.append(response.choices[0].message.content)
    return simplified_questions

def get_random_USMLE_questions_df():
    """Function: organizes a subset of randomly chosen USMLE questions into a DataFrame
    Return type: DataFrame"""
    questions = get_random_USMLE_questions()
    headers = ["Question"]
    df = pd.DataFrame(questions)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

def get_random_simplified_USMLE_questions_df():
    questions = get_random_simplified_USMLE_questions()
    headers = ["Question"]
    df = pd.DataFrame(questions)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

def get_stage_1_responses():
    """Function: retrieves responses to the subset of randomly chosen USMLE questions
    Return type: list"""
    questions = get_random_USMLE_questions()
    responses = []
    client = OpenAI()
    GPT_model = "gpt-4o"
    for question in questions:
        prompt = [{"role": "user", "content": question}]
        response = client.chat.completions.create(model = GPT_model, messages = prompt)
        responses.append(response.choices[0].message.content)
    return responses

def get_stage_1_simplified_responses():
    questions = get_random_simplified_USMLE_questions()
    responses = []
    client = OpenAI()
    GPT_model = "gpt-4o"
    for question in questions:
        prompt = [{"role": "user", "content": question}]
        response = client.chat.completions.create(model = GPT_model, messages = prompt)
        responses.append(response.choices[0].message.content)
    return responses

def get_stage_1_df():
    """Function: organizes the questions and responses from stage 1 into a DataFrame
    Return type: DataFrame"""
    stage_1_df = get_random_USMLE_questions_df()
    stage_1_responses = get_stage_1_responses()
    stage_1_df["Response"] = stage_1_responses
    return stage_1_df

def get_stage_1_simplified_df():
    stage_1_simplified_df = get_random_simplified_USMLE_questions_df()
    stage_1_simplified_responses = get_stage_1_simplified_responses()
    stage_1_simplified_df["Response"] = stage_1_simplified_responses
    return stage_1_simplified_df

def make_stage_1_csv(): 
    """Function: creates a CSV file containing the questions and responses from stage 1"""
    df = get_stage_1_df()
    df.to_csv("stage_1_GPT.csv", index = False)

def make_stage_1_simplified_csv():
    df = get_stage_1_simplified_df()
    df.to_csv("stage_1_simp_GPT.csv", index = False)

def get_USMLE_questions_with_race(): 
    """Function: generates USMLE questions with accompanying race demographics. Uses the 
    race categories provided on the U.S. Census.
    Return type: list of tuples"""
    race = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    USMLE_questions = get_random_USMLE_questions()
    questions_with_race = []
    for question in USMLE_questions:
        start_index = question.find("year-old")
        random_race = str(random.choices(race)[0])
        race_added = question[: start_index + 9] + random_race + " " + question[start_index + 9:]
        questions_with_race.append((random_race, race_added))
    return questions_with_race

def get_USMLE_questions_with_race_df():
    """Function: organizes USMLE questions with added race demographics into a DataFrame
    Return type: DataFrame"""
    questions = get_USMLE_questions_with_race()
    headers = ["Race", "Question"]
    df = pd.DataFrame(questions)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

def get_stage_2_race_responses():
    """Function: retrieves responses to the USMLE questions with added race demographics
    Return type: list"""
    questions = get_USMLE_questions_with_race()
    responses = []
    client = OpenAI()
    GPT_model = "gpt-4o"
    for question in questions:
        prompt = [{"role": "user", "content": str(question)}]
        response = client.chat.completions.create(model = GPT_model, messages = prompt)
        responses.append(response.choices[0].message.content)
    return responses

def get_stage_2_race_df():
    """Function: organizes the questions and responses from stage 2 with race into a DataFrame
    Return type: DataFrame"""
    stage_2_race_df = get_USMLE_questions_with_race_df()
    stage_2_race_responses = get_stage_2_race_responses()
    stage_2_race_df["Response"] = stage_2_race_responses
    return stage_2_race_df

def make_stage_2_race_csv(): 
    """Function: creates a CSV file containing the questions and responses from stage 2 with race"""
    df = get_stage_2_race_df()
    df.to_csv("stage_2_race_GPT.csv", index = False)

def get_USMLE_questions_with_income():
    """Function: generates USMLE questions with accompanying income demographics
    Return type: list of tuples"""
    income = ["lower class", "middle class", "upper class"]
    USMLE_questions = get_random_USMLE_questions()
    questions_with_income = []
    for question in USMLE_questions:
        start_index = question.find("year-old")
        random_income = str(random.choices(income)[0])
        income_added = question[: start_index + 9] + random_income + " " + question[start_index + 9:]        
        questions_with_income.append((random_income, income_added))
    return questions_with_income

def get_USMLE_questions_with_income_df():
    """Function: organizes USMLE questions with added income demographics into a DataFrame
    Return type: DataFrame"""
    questions = get_USMLE_questions_with_income()
    headers = ["Income", "Question"]
    df = pd.DataFrame(questions)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

def get_stage_2_income_responses():
    """Function: retrieves responses to the USMLE questions with added race demographics
    Return type: list"""
    questions = get_USMLE_questions_with_income()
    responses = []
    client = OpenAI()
    GPT_model = "gpt-4o"
    for question in questions:
        prompt = [{"role": "user", "content": str(question)}]
        response = client.chat.completions.create(model = GPT_model, messages = prompt)
        responses.append(response.choices[0].message.content)
    return responses

def get_stage_2_income_df():
    """Function: organizes the questions and responses from stage 2 with income into a DataFrame
    Return type: DataFrame"""
    stage_2_income_df = get_USMLE_questions_with_income_df()
    stage_2_income_responses = get_stage_2_income_responses()
    stage_2_income_df["Response"] = stage_2_income_responses
    return stage_2_income_df

def make_stage_2_income_GPT_csv(): 
    """Function: creates a CSV file containing the questions and responses from stage 2 with income"""
    df = get_stage_2_income_df()
    df.to_csv("stage_2_income_GPT.csv", index = False)

def get_general_public_USMLE_questions(): 
    """Function: generates a random subset of USMLE questions in the tone of the general public.
    Provides Reddit examples to the system role to train the LLM.
    Return type: list of tuples"""
    client = OpenAI()
    GPT_model = "gpt-4o"
    USMLE_questions = get_random_USMLE_questions()
    responses = []
    tone = "general public"
    for question in USMLE_questions:
        reddit_example_1 = "5 month old male, approx 16lbs. Possible milk allergy and GERD. Waiting on an allergist appointment in early July. Last night my 5 month old was asleep next to me in the bed around 8p, suddenly he started bringing his legs up to belly and arms perpendicular to body in like spams with 1-2 second pauses between each spasm. It last maybe 5-6 spasms and then he woke with hiccups immediately after stopping the spasms. He was acting normal afterwards. I messaged his pedi but haven’t heard back yet. I then was rocking him to sleep approx 10pm and he was doing this weird things with eyes and tightening his body for around 3 minutes before he finally fell asleep. I recorded it and have added link. I’m just not sure if this is being an overly anxious mom or if this is something that needs immediate attention. Thank you for all your help!"
        reddit_example_2 = "39yo female. 5’5” 135lb. I am experiencing jaw pain only on the left side. It started when my toddler son accidentally slammed his head into it two weeks ago. It was more jarring than painful when it happened. It’s only gotten worse instead of better. It’s not a constant pain but it’s hard to open my mouth all the way to eat. I’m also a stomach sleeper and it’s uncomfortable to sleep on my left side. My question is - what’s the best type of doctor to see for this? Thanks!"
        reddit_example_3 = "im 16F, 56kg was doing 120kg leg press at the gym earlier which is not a top set for me. at the bottom of my rep my hip stung a bit so i stopped after that rep. The outside of my right leg then went cold. Its 4 hours later and now it stings and the outside of my leg has gone completely numb. Like i cant feel it at all. Wtf is this. Can i still train legs??"
        prompt = [{"role": "system", "content": "You are translating professional medical questions into medical questions asked by the " + tone + " community. Here are some examples of the type of questions asked by the " + tone + " community. First example: " + reddit_example_1 + " Second example: " + reddit_example_2 + " Third example: " + reddit_example_3},{"role": "user", "content": "Translate the following question." + question}]
        response = client.chat.completions.create(model = GPT_model, messages = prompt)
        responses.append((tone, response.choices[0].message.content))
    return responses

def get_general_public_USMLE_questions_df():
    """Function: organizes general public USMLE questions into a DataFrame
    Return type: DataFrame"""
    questions = get_general_public_USMLE_questions()
    headers = ["Tone", "Question"]
    df = pd.DataFrame(questions)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

def get_stage_3_general_public_responses():
    """Function: retrieves responses to the general public USMLE questions
    Return type: list"""
    questions = get_general_public_USMLE_questions()
    responses = []
    client = OpenAI()
    GPT_model = "gpt-4o"
    for question in questions:
        prompt = [{"role": "user", "content": str(question[1])}]
        response = client.chat.completions.create(model = GPT_model, messages = prompt)
        responses.append(response.choices[0].message.content)
    return responses

def get_stage_3_general_public_df():
    """Function: organizes the questions and responses from stage 3 general public into a DataFrame
    Return type: DataFrame"""
    stage_3_df = get_general_public_USMLE_questions_df()
    stage_3_responses = get_stage_3_general_public_responses()
    stage_3_df["Response"] = stage_3_responses
    return stage_3_df

def make_stage_3_GPT_csv():
    """Function: creates a CSV file containing the tone, questions, and responses from stage 3 general public"""
    df = get_stage_3_general_public_df()
    df.to_csv("stage_3_general_public_GPT.csv", index = False)


#print(get_random_simplified_USMLE_questions())
#make_stage_1_simplified_csv()










