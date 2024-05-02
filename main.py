import os
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama
from crewai_tools import SerperDevTool
import requests
from requests.exceptions import ConnectionError
import time

os.environ["SERPER_API_KEY"] = "bc7eb598d6f80bfa8b6bd4313c2eb42bb964747b"

llm = Ollama(model="openhermes")

search_tool = SerperDevTool()

researcher = Agent(
    llm=llm,
    role="Senior Property Researcher",
    goal="Find promising investment properties.",
    backstory="You are a veteran property analyst. In this case, you're looking for retail properties to invest in.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=True,
)

task1 = Task(
    description="Search the internet and find 5 promising real estate investment suburbs in Rajasthan, India. For each suburb, highlight the mean, low and max prices as well as the rental yield and any potential factors that would be useful to know for that area.",
    expected_output="""A detailed report of each of the suburbs. The results should be formatted as shown below: 

    Suburb 1: Jodhpur
    Mean Price: $200,000
    Rental Vacancy: 5%
    Rental Yield: 2%
    Background Information: These suburbs are typically located near major harware industries , transport hubs and spices hubs, employment centers, and educational institutions. The following list highlights some of the top contenders for investment opportunities""",
    agent=researcher,
    output_file="task1_output.txt",
)

writer = Agent(
    llm=llm,
    role="Senior Property Analyst",
    goal="Summarise property facts into a report for investors.",
    backstory="You are a real estate agent, your goal is to compile property analytics into a report for potential investors.",
    allow_delegation=False,
    verbose=True,
)

task2 = Task(
    description="Summarise the property information into a bullet point list.",
    expected_output="A summarised dot point list of each of the suburbs, prices, and important features of that suburb.",
    agent=writer,
    output_file="task2_output.txt",
)

crew = Crew(agents=[researcher,writer], tasks=[task1, task2], verbose=2)

task_output = crew.kickoff()
print(task_output)