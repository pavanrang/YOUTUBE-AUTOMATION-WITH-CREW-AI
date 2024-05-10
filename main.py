from crewai import Crew, Process
from agents import YoutubeAutomationAgents
from tasks import YoutubeAutomationTasks
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from tools.youtube_video_details_tool import YoutubeVideoDetailsTool
from tools.youtube_video_search_tool import YoutubeVideoSearchTool

# Initialize the Groq Mistral language model using Groq Base URL:
openAI_groq_mistral = ChatOpenAI(
   base_url='https://api.groq.com/openai/v1',
    model="mixtral-8x7b-32768",
    # model="llama2-70b-4096",
    # model="gemma-7b-it",
    api_key='gsk_xQa4mRS2YhaQHbDbVvZTWGdyb3FY0.......'
)


agents = YoutubeAutomationAgents()
tasks = YoutubeAutomationTasks()

youtube_video_search_tool = YoutubeVideoSearchTool()
youtube_video_details_tool = YoutubeVideoDetailsTool()

youtube_manager = agents.youtube_manager()
research_manager = agents.research_manager(
    youtube_video_search_tool, youtube_video_details_tool)
title_creator = agents.title_creator()
description_creator = agents.description_creator()

video_topic = "How to get more subscribers on YouTube"
video_details = """
In this video, we're delving into the strategies and techniques to grow your 
YouTube subscriber count exponentially. I'll be sharing insider tips on 
leveraging social media platforms effectively, optimizing your video content 
for maximum engagement, and implementing proven methods to attract new 
subscribers organically. Join me as I unveil the secrets to building a 
loyal and dedicated subscriber base that will propel your YouTube channel 
to new heights of success.
"""


manage_youtube_video_creation = tasks.manage_youtube_video_creation(
    agent=youtube_manager,
    video_topic=video_topic,
    video_details=video_details
)
manage_youtube_video_research = tasks.manage_youtube_video_research(
    agent=research_manager,
    video_topic=video_topic,
    video_details=video_details,
)
create_youtube_video_title = tasks.create_youtube_video_title(
    agent=title_creator,
    video_topic=video_topic,
    video_details=video_details
)
create_youtube_video_description = tasks.create_youtube_video_description(
    agent=description_creator,
    video_topic=video_topic,
    video_details=video_details
)


# Create a new Crew instance
crew = Crew(
    agents=[youtube_manager,
            research_manager,
            title_creator,
            description_creator
            ],

    tasks=[manage_youtube_video_creation,
           manage_youtube_video_research,
           create_youtube_video_title,
           create_youtube_video_description],

    process=Process.sequential,
    manager_llm=openAI_groq_mistral
)

# Kick of the crew
results = crew.kickoff()

print("Crew usage", crew.usage_metrics)

print("Crew work results:")
print(results)
