import random
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import scipy.stats as stats
import statistics
import matplotlib.pyplot as plot

"""Link to Google Doc with all testing results: 
https://docs.google.com/document/d/1KtUQgn-0Bhv7B0Sh2bZP1qmB7tfojfh6W0vp8fkvQPU/edit#heading=h.k7k658iqty7k"""

def read_data():
    """Reads in USMLE data from HuggingFace and returns it in a DataFrame.
    Link to dataset: https://huggingface.co/datasets/augtoma/medqa_usmle"""
    all_data = {"train": "data/train-00000-of-00001-a6e6f83961340098.parquet", 
                "test": "data/test-00000-of-00001-997553f5b24c9767.parquet"}
    df = pd.read_parquet("hf://datasets/augtoma/medqa_usmle/" + all_data["train"])
    return df

def read_questions():
    """Extracts the questions from the USMLE data and returns them in a list."""
    df = read_data()
    questions = []
    for index, row in df.iterrows():
        question = row["question"]
        questions.append(question)
    return questions

def read_answers():
    """Extracts the ground-truth answers from the USMLE data and returns them in a list."""
    df = read_data()
    answers = []
    for index, row in df.iterrows():
        answer = row["answer"]
        answers.append(answer)
    return answers

def get_random(number, data):
    """Given some data and a desired size, returns a randomly selected subset of the data of the 
    desired size. The random seed can be changed to change the subset of data that is selected."""
    random.seed(48)
    result = random.choices(data, k = number)
    return result

def get_df(data, headers):
    """Given a list of data, returns the data in a DataFrame."""
    df = pd.DataFrame(data)
    df.columns = headers
    df.insert(0, "Index", df.index)
    return df

def add_column_to_df(df, data, header):
    """Adds a column (including data and header) to a pre-existing DataFrame."""
    df[header] = data

def get_results_df(questions, responses, answers, accuracy): 
    """Returns a DataFrame containing questions, GPT-3.5's responses, ground-truth answers, and 
    GPT-3.5-determined accuracy ratings of the responses."""
    df = get_df(questions, ["Question"])
    add_column_to_df(df, responses, "Response")
    add_column_to_df(df, answers, "Answer")
    add_column_to_df(df, accuracy, "Accuracy")
    return df

def make_csv(df, filename):
    """Given a DataFrame, converts the DataFrame into a CSV file."""
    df.to_csv(filename, index = False)

def query_GPT_without_system_role(user_prompt):
    """Submits a query to GPT-3.5 without a system role. Returns GPT-3.5's response."""
    client = OpenAI()
    model = "gpt-3.5-turbo-0125"
    prompt = [{"role": "user", "content": user_prompt}]
    response = client.chat.completions.create(model = model, messages = prompt)
    return response.choices[0].message.content

def query_GPT_with_system_role(system_prompt, user_prompt):
    """Submits a query to GPT-3.5 with a system role. Returns GPT-3.5's response."""
    client = OpenAI()
    model = "gpt-3.5-turbo-0125"
    prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = client.chat.completions.create(model = model, messages = prompt)
    return response.choices[0].message.content

def get_responses_without_system_role(questions):
    """For each question, generates a response without system role from GPT-3.5 and stores it in a list."""
    results = []
    for question in questions:
        response = query_GPT_without_system_role(question)
        results.append(response)
    return results 

def get_responses_with_system_role(questions, system_prompt):
    """For each question, generates a response with system role from GPT-3.5 and stores it in a list."""
    results = []
    for question in questions:
        response = query_GPT_with_system_role(system_prompt, question)
        results.append(response)
    return results

def run_accuracy_test(answers, responses):
    """Given a response and a ground-truth answer, submits a query to GPT-3.5 to determine if the response is 
    accurate, relevant, or inaccurate compared to the ground-truth answer. Stores the accuracy ratings in a list."""
    results = []
    index = 0
    system_prompt = """You will be given a ground truth answer and a response to a medical question. The response
                    will contain the three most plausible answers to the question. Compared to the ground truth answer,
                    if at least one of the three answers provided is accurate, respond with accurate. If none of the three
                    answers provided are accurate but at least one of them is relevant to the ground truth answer, respond
                    with relevant. If none of the three answers provided are accurate nor relevant compared to the ground
                    truth answer, respond with inaccurate. Respond only with accurate, relevant, or inaccurate. Do not
                    explain your reasoning."""
    for response in responses:
        user_prompt = "Ground truth answer: " + answers[index] + ". Response: " + response
        result = query_GPT_with_system_role(system_prompt, user_prompt)
        results.append(result)
        index = index + 1
    return results

def get_accuracy_statistics(df):
    """Given a DataFrame with a column for accuracy ratings, calculates the proportion of accurate, relevant, 
    and inaccurate ratings."""
    total_responses = 0
    accurate_responses = 0
    relevant_responses = 0
    inaccurate_responses = 0
    for index, row in df.iterrows():
        total_responses = total_responses + 1
        data = row["Accuracy"]
        accuracy = data.replace(".", "")
        if accuracy == "Accurate" or accuracy == "accurate":
            accurate_responses = accurate_responses + 1
        elif accuracy == "Relevant" or accuracy == "relevant":
            relevant_responses = relevant_responses + 1
        elif accuracy == "Inaccurate" or accuracy == "inaccurate":
            inaccurate_responses = inaccurate_responses + 1
    accurate = round(accurate_responses / total_responses, 2)
    relevant = round(relevant_responses / total_responses, 2)
    inaccurate = round(inaccurate_responses / total_responses, 2)
    results = ["Accurate:", accurate, "Relevant:", relevant, "Inaccurate", inaccurate]
    return results

def present_results(questions, filename):
    """Presents the results from feeding GPT-3.5 a certain type of medical question. Creates a CSV file with
    the newly generated results DataFrame and also prints out a summary of the calculated accuracy ratings."""
    system_prompt = """You will be given a medical question. Provide the top three most plausible answers to the question.
                    Number the answers clearly with *1* corresponding to the best answer, *2* corresponding to the second best
                    answer, and *3* corresponding to the third best answer. """
    responses = get_responses_with_system_role(questions, system_prompt)
    answers = get_random(50, read_answers())
    accuracy = run_accuracy_test(answers, responses)
    df = get_results_df(questions, responses, answers, accuracy)
    make_csv(df, filename)
    accuracy_statistics = get_accuracy_statistics(df)
    print(filename, accuracy_statistics)

def get_age_gender_questions():
    """BASELINE QUESTIONS. Returns a randomly selected subset of 50 questions from the USMLE data."""
    return get_random(50, read_questions())

def get_age_questions():
    """Returns a randomly selected subset of 50 USMLE questions with age removed."""
    original_questions = get_random(50, read_questions())
    system_prompt = "Rewrite the question provided with age removed."
    questions = get_responses_with_system_role(original_questions, system_prompt)
    return questions

def get_gender_questions():
    "Returns a randomly selected subset of 50 USMLE questions with gender removed."
    original_questions = get_random(50, read_questions())
    system_prompt = "Rewrite the question provided with gender removed. Change pronouns to be gender neutral."
    questions = get_responses_with_system_role(original_questions, system_prompt)
    return questions

def get_race_questions():
    """Returns a randomly selected subset of 50 USMLE questions with age and gender removed and a randomly
    chosen race added."""
    questions = []
    original_questions = get_random(50, read_questions())
    races = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    for question in original_questions:
        random_race = str(random.choices(races)[0])
        system_prompt = """Rewrite the question provided with age and gender removed. Change pronouns to be gender netural.
                        Add that the person's race is """ + random_race + """."""
        response = query_GPT_with_system_role(system_prompt, question)
        questions.append(response)
    return questions

def get_age_race_questions():
    """Returns a randomly selected subset of 50 USMLE questions with age and a randomly chosen race included."""
    questions = []
    original_questions = get_random(50, read_questions())
    races = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    for question in original_questions:
        random_race = str(random.choices(races)[0])
        system_prompt = """Rewrite the question provided with gender removed. Change pronouns to be gender netural.
                        Add that the person's race is """ + random_race + """."""
        response = query_GPT_with_system_role(system_prompt, question)
        questions.append(response)
    return questions

def get_gender_race_questions():
    """Returns a randomly selected subset of 50 UMSLE questions with gender and a randomly chosen race included."""
    questions = []
    original_questions = get_random(50, read_questions())
    races = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    for question in original_questions:
        random_race = str(random.choices(races)[0])
        system_prompt = """Rewrite the question provided with age removed. Add that the person's 
                        race is """ + random_race + """."""
        response = query_GPT_with_system_role(system_prompt, question)
        questions.append(response)
    return questions

def get_age_gender_race_questions():
    """Returns a randomly selected subset of 50 UMSLE questions with age, gender, and a randomly chosen race included."""
    questions = []
    original_questions = get_random(50, read_questions())
    races = ["American Indian/Alaska Native", "Asian", "Black/African American", "Native Hawaiian/Pacific Islander", "White"]
    for question in original_questions:
        random_race = str(random.choices(races)[0])
        system_prompt = """Rewrite the question provided to add that the person's race is """ + random_race + """."""
        response = query_GPT_with_system_role(system_prompt, question)
        questions.append(response)
    return questions

def get_reddit_questions():
    """Returns a randomly selected subset of 50 UMSLE questions phrased in the tone of a Reddit user."""
    questions = []
    original_questions = get_random(50, read_questions())
    first_reddit_example = """5 month old male, approx 16lbs. Possible milk allergy and GERD. Waiting on an allergist 
                            appointment in early July. Last night my 5 month old was asleep next to me in the bed around 8p, 
                            suddenly he started bringing his legs up to belly and arms perpendicular to body in like spams with 
                            1-2 second pauses between each spasm. It last maybe 5-6 spasms and then he woke with hiccups 
                            immediately after stopping the spasms. He was acting normal afterwards. I messaged his pedi but 
                            haven’t heard back yet. I then was rocking him to sleep approx 10pm and he was doing this weird things 
                            with eyes and tightening his body for around 3 minutes before he finally fell asleep. I recorded it and 
                            have added link. I’m just not sure if this is being an overly anxious mom or if this is something that 
                            needs immediate attention. Thank you for all your help!"""
    second_reddit_example = """39yo female. 5’5” 135lb. I am experiencing jaw pain only on the left side. It started when my toddler 
                            son accidentally slammed his head into it two weeks ago. It was more jarring than painful when it 
                            happened. It’s only gotten worse instead of better. It’s not a constant pain but it’s hard to open my 
                            mouth all the way to eat. I’m also a stomach sleeper and it’s uncomfortable to sleep on my left side. 
                            My question is - what’s the best type of doctor to see for this? Thanks!"""
    third_reddit_example = """28M, had surgery yesterday. This was my 4th surgery and usually when they take the IV out of my hand 
                            afterwords, they gently remove it. But yesterday my nurse just seemed kinda off and when she went to 
                            take my IV out, she quickly ripped it off backwards with the tape. It hurt and bled a lot but she didn’t 
                            say anything and just applied pressure for awhile. Just wondering if that’s normal or if that can cause 
                            any damage to my vein or anything like that? It still hurts a bit today and the area around it is kinda 
                            red."""
    system_prompt = """You are translating professional medical questions into medical questions asked by the general public.
                    Here are some examples of the types of questions asked by the general public. First example: 
                    """ + first_reddit_example + """Second example: """ + second_reddit_example + """Third example: 
                    """ + third_reddit_example + """. Given these examples, remove all medical terms in the question and replace
                    them with casual terms someone would use in everyday speech. Formulate the question as if someone without 
                    any medical expertise wrote it."""
    for question in original_questions:
        user_prompt = "Translate the following question: " + question + ". Respond only with the question."
        response = query_GPT_with_system_role(system_prompt, user_prompt)
        questions.append(response)
    return questions

def get_young_adult_questions():
    """Returns a randomly selected subset of 50 USMLE questions phrased in the tone of a young adult."""
    questions = []
    original_questions = get_random(50, read_questions())
    system_prompt = """You are translating professional medical questions into medical questions asked by a young adult.
                    People from this demographic are often informal, use casual language, and may express concerns or 
                    anxieties directly. They might use slang or digital communication abbreviations. Keep any lab
                    results that are given in the original professional medical question."""
    for question in original_questions:
        user_prompt = "Translate the following question: " + question + ". Respond only with the question."
        response = query_GPT_with_system_role(system_prompt, user_prompt)
        questions.append(response)
    return questions

def get_no_formal_education_questions():
    """Returns a randomly selected subet of 50 USMLE questions phrased in the tone of someone with no formal education."""
    questions = []
    original_questions = get_random(50, read_questions())
    system_prompt = """You are translating professional medical questions into medical questions asked by someone with no
                    formal education. People from this demographic have decreased oral comprehension, oral narrative discourse, 
                    and number reading skills. Keep any lab results that are given in the original professional medical
                    question."""
    for question in original_questions:
        user_prompt = "Translate the following question: " + question + ". Respond only with the question."
        response = query_GPT_with_system_role(system_prompt, user_prompt)
        questions.append(response)
    return questions

def get_non_native_speaker_questions():
    """Returns a randomly selected subset of 50 USMLE questions phrased in the tone of a non-Native English speaker."""
    questions = []
    original_questions = get_random(50, read_questions())
    system_prompt = """You are translating professional medical questions into medical questions asked by someone whose first language
            is not English. They speak English, but not perfectly. Add a few pauses to the rhythm of their speech.
            Make sure to keep any lab results or relevant drug names that are provided in the original professional medical 
            question in the translated question. Answer in English only."""
    for question in original_questions:
        user_prompt = "Translate the following question: " + question + ". Respond only with the question."
        response = query_GPT_with_system_role(system_prompt, user_prompt)
        questions.append(response)
    return questions

def get_p_value(stat_1, stat_2):
    """Given two statistics, returns the the p-value of the relationship between the two variables."""
    p_value = round(stats.ttest_rel(stat_1, stat_2).pvalue, 6)
    return p_value

def get_standard_deviation(data):
    """Given a set of data, returns the standard deviation of the data."""
    sd = round(statistics.stdev(data), 3)
    return sd

def get_95_percent_confidence_interval(mean, standard_deviation):
    """Given the mean and standard deviation of a set of data, returns the 95% confidence interval
    of the set of data."""
    z_score = 1.96
    lower_bound = round(mean - (z_score * standard_deviation), 3)
    upper_bound = round(mean + (z_score * standard_deviation), 3)
    confidence_interval = (lower_bound, upper_bound)
    return confidence_interval

def present_p_value_summary_demographic_tones(baseline_data, young_adult_data, reddit_data, no_formal_education_data, non_native_speaker_data):
    """Prints a summary of the p-values of the demographic tone results."""
    summary = []
    young_adult = get_p_value(baseline_data, young_adult_data)
    summary.append("Young adult: " + str(young_adult))
    reddit = get_p_value(baseline_data, reddit_data)
    summary.append("Reddit user: " + str(reddit))
    no_formal_education = get_p_value(baseline_data, no_formal_education_data)
    summary.append("No formal education: " + str(no_formal_education))
    non_native_speaker = get_p_value(baseline_data, non_native_speaker_data)
    summary.append("Non-Native English speaker: " + str(non_native_speaker))
    for stat in summary:
        print(stat)

def present_p_value_summary_demographic_factors(baseline_data, age_data, gender_data):
    """Prints a summary of the p-values of the demographic factor results."""
    summary = []
    age = get_p_value(baseline_data, age_data)
    summary.append("Age removed: " + str(age))
    gender = get_p_value(baseline_data, gender_data)
    summary.append("Gender removed: " + str(gender))
    for stat in summary:
        print(stat)

def present_sd_summary_demographic_tones(baseline_data, young_adult_data, reddit_data, no_formal_education_data, non_native_speaker_data):
    """Prints a summary of the standard deviations of the demographic tone results."""
    summary = []
    baseline = get_standard_deviation(baseline_data)
    summary.append("Baseline: " + str(baseline))
    young_adult = get_standard_deviation(young_adult_data)
    summary.append("Young adult: " + str(young_adult))
    reddit = get_standard_deviation(reddit_data)
    summary.append("Reddit user: " + str(reddit))
    no_formal_education = get_standard_deviation(no_formal_education_data)
    summary.append("No formal education: " + str(no_formal_education))
    non_native_speaker = get_standard_deviation(non_native_speaker_data)
    summary.append("Non-Native English speaker: " + str(non_native_speaker))
    for stat in summary:
        print(stat)

def present_sd_summary_demographic_factors(baseline_data, age_data, gender_data):
    """Prints a summary of the standard deviations of the demographic factor results."""
    summary = []
    baseline = get_standard_deviation(baseline_data)
    summary.append("Baseline: " + str(baseline))
    age = get_standard_deviation(age_data)
    summary.append("Age removed: " + str(age))
    gender = get_standard_deviation(gender_data)
    summary.append("Gender removed: " + str(gender))
    for stat in summary:
        print(stat)
    
def present_mean_summary_demographic_tones(baseline_data, young_adult_data, reddit_data, no_formal_education_data, non_native_speaker_data):
    """Prints a summary of the mean accuracies of the demographic tone results."""
    summary = []
    baseline = statistics.mean(baseline_data)
    summary.append("Baseline: " + str(baseline))
    young_adult = statistics.mean(young_adult_data)
    summary.append("Young adult: " + str(young_adult))
    reddit = statistics.mean(reddit_data)
    summary.append("Reddit user: " + str(reddit))
    no_formal_education = statistics.mean(no_formal_education_data)
    summary.append("No formal education: " + str(no_formal_education))
    non_native_speaker = statistics.mean(non_native_speaker_data)
    summary.append("Non-Native English speaker: " + str(non_native_speaker))
    for stat in summary:
        print(stat)

def present_mean_summary_demographic_factors(baseline_data, age_data, gender_data):
    """Prints a summary of the mean accuracies of the demographic factor results."""
    summary = []
    baseline = statistics.mean(baseline_data)
    summary.append("Baseline: " + str(baseline))
    age = statistics.mean(age_data)
    summary.append("Age removed: " + str(age))
    gender = statistics.mean(gender_data)
    summary.append("Gender removed: " + str(gender))
    for stat in summary:
        print(stat)

def present_95_percent_confidence_interval_summary_demographic_tones(baseline_data, young_adult_data, reddit_data, no_formal_education_data, non_native_speaker_data):
    """Prints a summary of the 95% confidence intervals of the demographic tone results."""
    summary = []
    baseline = get_95_percent_confidence_interval(statistics.mean(baseline_data), get_standard_deviation(baseline_data))
    summary.append("Baseline: " + str(baseline))
    young_adult = get_95_percent_confidence_interval(statistics.mean(young_adult_data), get_standard_deviation(young_adult_data))
    summary.append("Young adult: " + str(young_adult))
    reddit = get_95_percent_confidence_interval(statistics.mean(reddit_data), get_standard_deviation(reddit_data))
    summary.append("Reddit user: " + str(reddit))
    no_formal_education = get_95_percent_confidence_interval(statistics.mean(no_formal_education_data), get_standard_deviation(no_formal_education_data))
    summary.append("No formal education: " + str(no_formal_education))
    non_native_speaker = get_95_percent_confidence_interval(statistics.mean(non_native_speaker_data), get_standard_deviation(non_native_speaker_data))
    summary.append("Non-Native English speaker: " + str(non_native_speaker))
    for stat in summary:
        print(stat)

def present_95_percent_confidence_interval_summary_demographic_factors(baseline_data, age_data, gender_data):
    """Prints a summary of the 95% confidence intervals of the demographic factor results."""
    summary = []
    baseline = get_95_percent_confidence_interval(statistics.mean(baseline_data), get_standard_deviation(baseline_data))
    summary.append("Baseline: " + str(baseline))
    age = get_95_percent_confidence_interval(statistics.mean(age_data), get_standard_deviation(age_data))
    summary.append("Age removed: " + str(age))
    gender = get_95_percent_confidence_interval(statistics.mean(gender_data), get_standard_deviation(gender_data))
    summary.append("Gender removed: " + str(gender))
    for stat in summary:
        print(stat)

def present_single_boxplot(data, lower_y_bound, upper_y_bound, x_label, title):
    """Plots a single box plot with desired input, bounds, x-axis tick mark label, and title."""
    plot.figure(figsize = (10, 7))
    plot.ylim(lower_y_bound, upper_y_bound)
    plot.boxplot(data, medianprops = dict(color = "black"))
    plot.title(title, weight = "bold")
    plot.xlabel("X-Axis Label", weight = "bold")
    plot.xticks([1], [x_label])
    plot.ylabel("Y-Axis Label", weight = "bold")
    plot.show()

def present_significant_boxplots_demographic_tones(baseline, reddit, no_formal_education, non_native_speaker):
    """Presents a figure containing box plots of the statistically significant demographic tone results.
    The young adult tone did not prove to be statistically significant so it is not included."""
    plot.figure(figsize = (10, 7))
    plot.rcParams["font.size"] = 13.5
    plot.ylim(0.51, 0.93)
    plot.boxplot([baseline, non_native_speaker, reddit, no_formal_education], medianprops = dict(color = "black"))
    plot.xlabel("Demographic Tone of Question", weight = "bold")
    plot.xticks([1, 2, 3, 4], ["Professional", "Non-Native English Speaker", "Reddit", "No Formal Education"])
    plot.ylabel("GPT-3.5 Accuracy", weight = "bold")
    plot.title("GPT-3.5 Accuracy vs. Demographic-Specific Tones", weight = "bold")
    plot.show()

def present_boxplots_demographic_factors(baseline, age, gender):
    """Presents a figure containing box plots of the demographic factor results. The results are not statistically significant."""
    plot.figure(figsize = (10,7))
    plot.rcParams["font.size"] = 13.5
    plot.ylim(0.63, 0.93)
    plot.boxplot([baseline, gender, age], medianprops = dict(color = "black"))
    plot.xlabel("Question Type", weight = "bold")
    plot.xticks([1, 2, 3], ["Professional", "Gender Removed", "Age Removed"])
    plot.ylabel("GPT-3.5 Accuracy", weight = "bold")
    plot.title("GPT-3.5 Accuracy vs. Removal of Demographic Factors", weight = "bold")
    plot.show()

def present_multiple_results_with_parallel_processing():
    """Processes and presents multiple indpendent functions simultaneously using threads. You can change the input 
    functions below depending on which results you want to retrieve. The resulting CSV will be stored in the file name 
    given in the second position of each tuple below."""
    inputs = [(get_age_gender_questions(), "baseline.csv"), (get_reddit_questions(), "reddit.csv"), (get_no_formal_education_questions(), "no_formal_education.csv"), (get_non_native_speaker_questions(), "non_native_speaker.csv")]
    with ThreadPoolExecutor() as executor:
        executor.map(lambda f: present_results(*f), inputs)

def main():
    """These are the mean accuracy results for each factor we tested (10 trials in total). DO NOT DELETE."""
    baseline_accuracy = [0.72, 0.78, 0.80, 0.92, 0.78, 0.86, 0.74, 0.70, 0.84, 0.84]
    age_accuracy = [0.72, 0.78, 0.70, 0.86, 0.76, 0.76, 0.80, 0.64, 0.84, 0.80]
    gender_accuracy = [0.64, 0.86, 0.84, 0.84, 0.82, 0.76, 0.72, 0.72, 0.88, 0.82]
    reddit_accuracy = [0.64, 0.62, 0.80, 0.76, 0.52, 0.76, 0.64, 0.70, 0.66, 0.64]
    young_adult_accuracy = [0.78, 0.82, 0.76, 0.78, 0.68, 0.86, 0.80, 0.64, 0.76, 0.68]
    no_formal_education_accuracy = [0.74, 0.66, 0.76, 0.72, 0.68, 0.72, 0.60, 0.62, 0.70, 0.74]
    non_native_speaker_accuracy = [0.80, 0.76, 0.72, 0.80, 0.72, 0.62, 0.72, 0.72, 0.76, 0.76]
    
    """Uncomment below to run multiple independent tests at the same time."""
    #present_multiple_results_with_parallel_processing()

    """Uncomment below to retrieve p-value summaries."""
    #present_p_value_summary_demographic_tones(baseline_accuracy, young_adult_accuracy, reddit_accuracy, no_formal_education_accuracy, non_native_speaker_accuracy)
    #present_p_value_summary_demographic_factors(baseline_accuracy, age_accuracy, gender_accuracy)

    """Uncomment below to retrieve mean accuracy summaries."""
    #present_mean_summary_demographic_tones(baseline_accuracy, young_adult_accuracy, reddit_accuracy, no_formal_education_accuracy, non_native_speaker_accuracy)
    #present_mean_summary_demographic_factors(baseline_accuracy, age_accuracy, gender_accuracy)

    """Uncomment below to retrieve standard deviation summaries."""
    #present_sd_summary_demographic_tones(baseline_accuracy, young_adult_accuracy, reddit_accuracy, no_formal_education_accuracy, non_native_speaker_accuracy)
    #present_sd_summary_demographic_factors(baseline_accuracy, age_accuracy, gender_accuracy)

    """Uncomment below to retrieve 95% confidence interval summaries."""
    #present_95_percent_confidence_interval_summary_demographic_tones(baseline_accuracy, young_adult_accuracy, reddit_accuracy, no_formal_education_accuracy, non_native_speaker_accuracy)
    #present_95_percent_confidence_interval_summary_demographic_factors(baseline_accuracy, age_accuracy, gender_accuracy)

    """Uncomment below to present box plots."""
    #present_significant_boxplots_demographic_tones(baseline_accuracy, reddit_accuracy, no_formal_education_accuracy, non_native_speaker_accuracy)
    #present_boxplots_demographic_factors(baseline_accuracy, age_accuracy, gender_accuracy)
       
if __name__ == "__main__":
    main()

