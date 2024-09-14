import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.utilities import WikipediaAPIWrapper
from langchain.llms import HuggingFaceHub, GooglePalm
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import streamlit as st

from dotenv import load_dotenv, find_dotenv
load_dotenv()

#Initialise llm model
# llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha",model_kwargs={'temperature':0.1,'max-length':512})
llm = GooglePalm(model="gemini-pro",temperature=0.1)    

# initialise key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}
if 'df' not in st.session_state:
    st.session_state.df = None

st.title('🤖 Data science AI assistant')
st.write('Hello 👋 I am your AI assistant, how can I help you?')

with st.sidebar:
    st.header('**EDA for your data**')
    st.subheader('*Once you upload your CSV files I will be able to look into them and provide you with insights!*')

    # function to update the value in session state
    def clicked(button):
        st.session_state.clicked[button] = True

    user_csv = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv,low_memory=False)
        with open(user_csv.name, 'wb') as f:
            f.write(user_csv.getbuffer())
    st.button("Process", on_click=clicked, args=[1])
        
        

    st.divider()

    st.caption("<p style='text-align:center'>Made with love by me</p>",unsafe_allow_html=True)

#Functions main
@st.cache_data
def eda_agent():
    st.write("**Data Overview**")
    st.write("The columns of your dataset are:")
    columns_df = pandas_agent.run("What are the meaning of the columns, tell me line by line?")
    st.write(columns_df)
    st.write("The first rows of your dataset look like this:")
    st.write(df.head(10))
    st.write("**Data Cleaning**")
    missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
    st.write(missing_values)
    duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
    st.write(duplicates)
    st.write("**Data Summarisation**")
    st.write(df.describe())
    correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
    st.write(correlation_analysis)
    outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
    st.write(outliers)
    new_features = pandas_agent.run("What are other useful columns that this dataframe could have? Don't add them to the dataframe, just list them")
    st.write(new_features)
    return

@st.cache_data
def question_variable():
    column_name = pandas_agent.run(f"Give me just the exact column name that {user_question_variable} could refer to")
    st.line_chart(df, y =[column_name])
    summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
    st.write(summary_statistics)
    normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
    st.write(normality)
    outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
    st.write(outliers)
    trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
    st.write(trends)
    missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
    st.write(missing_values)
    return

def question_dataframe():
    dataframe_info = pandas_agent.run(user_question_dataframe)
    st.write(dataframe_info)
    return

@st.cache_resource
def problem_ML_models():
    data_problem_template = PromptTemplate(
        input_variables=['business_problem'],
        template='Convert the following business problem into a data science problem: {business_problem}.'
    )

    model_selection_template = PromptTemplate(
        input_variables=['data_problem', 'wikipedia_research'],
        template='Give just a list of machine learning algorithms and nothing else that are suitable to solve this problem: {data_problem}, while using these columns: {wikipedia_research}.'
    )
    data_problem_chain = LLMChain(llm=llm, prompt=data_problem_template, verbose=True, output_key='data_problem')
    model_selection_chain = LLMChain(llm=llm, prompt=model_selection_template, verbose=True, output_key='model_selection')
    sequential_chain = SequentialChain(chains=[data_problem_chain, model_selection_chain], input_variables=['business_problem', 'wikipedia_research'], output_variables=['data_problem', 'model_selection'], verbose=True)
    return sequential_chain

@st.cache_data
def list_to_selectbox(my_model_selection_input):
    algorithm_lines = my_model_selection_input.split('\n')
    algorithms = [algorithm.split(':')[-1].split('.')[-1].strip() for algorithm in algorithm_lines if algorithm.strip()]
    algorithms.insert(0, "Select Algorithm")
    formatted_list_output = [f"{algorithm}" for algorithm in algorithms if algorithm]
    return formatted_list_output

@st.cache_resource
def python_agent():
    agent_executor = create_python_agent(
        llm=llm,
        df=df,
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        )
    return agent_executor

@st.cache_data
def python_solution(my_data_problem, selected_algorithm, user_csv):
    solution = python_agent().run(
        f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {user_csv}."
    )
    return solution

if st.session_state.clicked[1]:
    st.header('EDA for your data')
    pandas_agent = create_pandas_dataframe_agent(llm, df, handle_parsing_errors=True, verbose=True)
    
    eda_agent()
    st.subheader('Variable of study')
    user_question_variable = st.text_input('What variable are you interested in')
    if user_question_variable is not None and user_question_variable !="":
        question_variable()

    st.subheader('Further study')
    user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
    if user_question_dataframe is not None and user_question_variable !="":
        question_dataframe()

    st.subheader('Data science problem')
    
    prompt = st.text_area('What is your business problem?')


    if prompt:
        list_of_cols = df.columns.tolist()
        cols = "".join(col+" " for col in list_of_cols)
        # keywords = llm(f"Generate space seperated keywords for a wikipedia search to help choose an ML model for this problem: {prompt} given {cols}")
        # wiki_research = WikipediaAPIWrapper().run(keywords)
        wiki_research = cols
        problem_ML_chain = problem_ML_models()
        problem_ML_response = problem_ML_chain({'business_problem': prompt, 'wikipedia_research': wiki_research})

        my_data_problem = problem_ML_response['data_problem']
        my_model_selection = problem_ML_response['model_selection']
                            
        st.write(my_data_problem)
        st.write(my_model_selection)

        formatted_list = list_to_selectbox(my_model_selection)
        selected_algorithm = st.selectbox("Select Machine Learning Algorithm", formatted_list)

        if selected_algorithm is not None and selected_algorithm != "Select Algorithm":
            st.subheader("Solution")
            solution = python_solution(my_data_problem, selected_algorithm, user_csv)
            st.write(solution)