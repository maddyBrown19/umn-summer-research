import random
import pandas as pd
from openai import OpenAI


"""
TODO this week (and next Monday):
- Generate baseline data and accuracy test: DONE
- Generate age data and accuracy test: DONE
- Generate gender data and accuracy test: DONE
- Generate race data and accuracy test: DONE
- Generate age/gender (original) data and accuracy test: DONE
- Generate age/race data and accuracy test: DONE
- Generate gender/race data and accuracy test: DONE
- Generate age/gender/race data and accuracy test: DONE
- Calculate accuracy statistics (percent accurate) for each test:
- Run all the same tests on different testing data (change random seed):
- Incorporate human (manual) evaluation into accuracy scores:
- Generate tone specific (general public, etc.) data and accuracy tests:

TODO when have time:
- Add function comments:
"""
def read_data():
    all_data = {"train": "data/train-00000-of-00001-a6e6f83961340098.parquet", 
                "test": "data/test-00000-of-00001-997553f5b24c9767.parquet"}
    df = pd.read_parquet("hf://datasets/augtoma/medqa_usmle/" + all_data["train"])
    return df

def read_questions():
    df = read_data()
    questions = []
    for index, row in df.iterrows():
        question = row["question"]
        questions.append(question)
    return questions

def read_answers():
    df = read_data()
    answers = []
    for index, row in df.iterrows():
        answer = row["answer"]
        answers.append(answer)
    return answers

def get_random(number, data):
    random.seed(10)
    result = random.choices(data, k = number)
    return result

def get_df(data, headers):
    df = pd.DataFrame(data)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

def add_column_to_df(df, data, header):
    df[header] = data

def get_results_df(questions, responses, answers, accuracy):
    df = get_df(questions, ["Question"])
    add_column_to_df(df, responses, "Response")
    add_column_to_df(df, answers, "Answer")
    add_column_to_df(df, accuracy, "Accurate")
    return df

def make_csv(df, filename):
    df.to_csv(filename, index = False)

def query_GPT_without_system_role(user_prompt):
    client = OpenAI()
    model = "gpt-4o"
    prompt = [{"role": "user", "content": user_prompt}]
    response = client.chat.completions.create(model = model, messages = prompt)
    return response.choices[0].message.content

def query_GPT_with_system_role(system_prompt, user_prompt):
    client = OpenAI()
    model = "gpt-4o"
    prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = client.chat.completions.create(model = model, messages = prompt)
    return response.choices[0].message.content

def get_responses_without_system_role(questions):
    results = []
    for question in questions:
        response = query_GPT_without_system_role(question)
        results.append(response)
    return results 

def get_responses_with_system_role(questions, system_prompt):
    results = []
    for question in questions:
        response = query_GPT_with_system_role(system_prompt, question)
        results.append(response)
    return results

def run_accuracy_test(answers, responses):
    results = []
    index = 0
    for response in responses:
        system_prompt = """You will be given a reference answer and an explanation of a medical question. Your job 
                        is to determine if the explanation contains the same underlying medical information as the 
                        reference answer. Respond with yes or no. Do not explain your reasoning."""
        user_prompt = "Reference answer: " + answers[index] + ". Explanation: " + response
        result = query_GPT_with_system_role(system_prompt, user_prompt)
        results.append(result)
        index = index + 1
    return results

def run_similarity_test():
    return

def get_baseline_questions():
    original_questions = get_random(20, read_questions())
    system_prompt = "Rewrite the question provided with age and gender removed. Change pronouns to be gender neutral."
    questions = get_responses_with_system_role(original_questions, system_prompt)
    return questions

def present_baseline_results():
    questions = get_baseline_questions()
    responses = get_responses_without_system_role(questions)
    answers = get_random(20, read_answers())
    accuracy = run_accuracy_test(answers, responses)
    df = get_results_df(questions, responses, answers, accuracy)
    make_csv(df, "baseline.csv")

def get_age_questions():
    original_questions = get_random(20, read_questions())
    system_prompt = "Rewrite the question provided with gender removed. Change pronouns to be gender neutral."
    questions = get_responses_with_system_role(original_questions, system_prompt)
    return questions

def present_age_results():
    questions = get_age_questions()
    responses = get_responses_without_system_role(questions)
    answers = get_random(20, read_answers())
    accuracy = run_accuracy_test(answers, responses)
    df = get_results_df(questions, responses, answers, accuracy)
    make_csv(df, "age.csv")

def get_gender_questions():
    original_questions = get_random(20, read_questions())
    system_prompt = "Rewrite the question provided with age removed."
    questions = get_responses_with_system_role(original_questions, system_prompt)
    return questions

def present_gender_results():
    questions = get_gender_questions()
    responses = get_responses_without_system_role(questions)
    answers = get_random(20, read_answers())
    accuracy = run_accuracy_test(answers, responses)
    df = get_results_df(questions, responses, answers, accuracy)
    make_csv(df, "gender.csv")

def get_race_questions():
    questions = []
    original_questions = get_random(20, read_questions())
    races = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    for question in original_questions:
        random_race = str(random.choices(races)[0])
        system_prompt = """Rewrite the question provided with age and gender removed. Change pronouns to be gender netural.
                        Add that the person's race is """ + random_race + """."""
        response = query_GPT_with_system_role(system_prompt, question)
        questions.append(response)
    return questions

def present_race_results():
    questions = get_race_questions()
    responses = get_responses_without_system_role(questions)
    answers = get_random(20, read_answers())
    accuracy = run_accuracy_test(answers, responses)
    df = get_results_df(questions, responses, answers, accuracy)
    make_csv(df, "race.csv")

def get_age_gender_questions():
    return get_random(20, read_questions())

def present_age_gender_results():
    questions = get_age_gender_questions()
    responses = get_responses_without_system_role(questions)
    answers = get_random(20, read_answers())
    accuracy = run_accuracy_test(answers, responses)
    df = get_results_df(questions, responses, answers, accuracy)
    make_csv(df, "age_gender.csv")

def get_age_race_questions():
    questions = []
    original_questions = get_random(20, read_questions())
    races = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    for question in original_questions:
        random_race = str(random.choices(races)[0])
        system_prompt = """Rewrite the question provided with gender removed. Change pronouns to be gender netural.
                        Add that the person's race is """ + random_race + """."""
        response = query_GPT_with_system_role(system_prompt, question)
        questions.append(response)
    return questions

def present_age_race_results():
    questions = get_age_race_questions()
    responses = get_responses_without_system_role(questions)
    answers = get_random(20, read_answers())
    accuracy = run_accuracy_test(answers, responses)
    df = get_results_df(questions, responses, answers, accuracy)
    make_csv(df, "age_race.csv")

def get_gender_race_questions():
    questions = []
    original_questions = get_random(20, read_questions())
    races = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    for question in original_questions:
        random_race = str(random.choices(races)[0])
        system_prompt = """Rewrite the question provided with age removed. Add that the person's 
                        race is """ + random_race + """."""
        response = query_GPT_with_system_role(system_prompt, question)
        questions.append(response)
    return questions

def present_gender_race_results():
    questions = get_gender_race_questions()
    responses = get_responses_without_system_role(questions)
    answers = get_random(20, read_answers())
    accuracy = run_accuracy_test(answers, responses)
    df = get_results_df(questions, responses, answers, accuracy)
    make_csv(df, "gender_race.csv")

def get_age_gender_race_questions():
    questions = []
    original_questions = get_random(20, read_questions())
    races = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    for question in original_questions:
        random_race = str(random.choices(races)[0])
        system_prompt = """Rewrite the question provided to add that the person's race is """ + random_race + """."""
        response = query_GPT_with_system_role(system_prompt, question)
        questions.append(response)
    return questions

def present_age_gender_race_results():
    questions = get_age_gender_race_questions()
    responses = get_responses_without_system_role(questions)
    answers = get_random(20, read_answers())
    accuracy = run_accuracy_test(answers, responses)
    df = get_results_df(questions, responses, answers, accuracy)
    make_csv(df, "age_gender_race.csv")

def main():
    present_age_gender_race_results()

if __name__ == "__main__":
    main()

