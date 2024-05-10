from crewai import Agent

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere


llm = ChatGroq(
            api_key="gsk_xQa4mRS2YhaQHbDbVvZTWGdyb3FY0J.......",
            model="mixtral-8x7b-32768"
)

# llm = ChatCohere(
#     cohere_api_key='9lmXAKqVY7o42K0NynBfAQ7d.......',
#     model="command-r-plus"
# )

# llm = ChatOpenAI(
#     model = "gpt-4-0314",
#     api_key="sk-a1b2c3d4e5f6g7h8i9j0k1l2m3n4............"
# )

# llm = ChatOpenAI(
#             base_url='https://api.groq.com/openai/v1',
#             model="mixtral-8x7b-32768",
#             api_key='gsk_xQa4mRS2YhaQHbDbVvZTWGdyb3FY0JI8U......'
#     )


class YoutubeAutomationAgents():
    def youtube_manager(self):
        return Agent(
            role="YouTube Manager",
            goal="Prepare a YouTube video by coordinating and managing the steps involved, which include market research, title generation, and description creation",

            backstory="""As a careful and organized manager, your job is to make sure YouTube videos are ready for upload. Here's how you do it:
                    1. Look on YouTube for at least 15 other videos about the same topic. Check out their titles and descriptions.
                    2. Come up with 10 possible titles for your video. Each title should be under 70 characters and really catchy. Give these titles to the person who makes the titles.
                    3. Write a description for your video.
                """,
            allow_delegation=True,
            verbose=True,
            llm=llm
        )

    def research_manager(self, youtube_video_search_tool, youtube_video_details_tool):
        return Agent(
            role="YouTube Research Manager",
            goal="""For a given topic and description for a new YouTube video, find a minimum of 15 high-performing videos 
                on the same topic with the ultimate goal of populating the research table which will be used by 
                other agents to help them generate titles  and other aspects of the new YouTube video 
                that we are planning to create.""",
            backstory="""Your job is to find and gather information about YouTube videos about the same topic, to help the creator make a great video. You'll be searching for videos, and retrieving details about them, so you can use that information to assist in other parts of the video creation process.""",
            verbose=True,
            allow_delegation=True,
            llm=llm,
            tools=[youtube_video_search_tool, youtube_video_details_tool]
        )

    def title_creator(self):
        return Agent(
            role="Title Creator",
            goal="""Your objective is to come up with 10 captivating and engaging title options for a given YouTube video. 
                To enhance your creativity, you may make use of the previous research you gathered from the YouTube Research Manager. 
                Each title should be concise (less than 70 characters), yet compelling and likely to prompt the viewer's interest and click-through-rate (CTR). 
                This CTR is crucial because high-performing videos typically have impressive CTRs, which can significantly boost your channel's visibility and engagement. 
                By delivering an array of attractive and potentially clickable titles, you contribute to the success of your YouTube video.""",

            backstory="""As a Title Creator, you are responsible for creating 10 potential titles for a given 
                YouTube video topic and description.""",
                
            verbose=True,
            llm = llm
        )

    def description_creator(self):
        return Agent(
            role="Description Creator",
            goal="""Your objective is to craft a compelling and engaging description that effectively captures the essence of the given YouTube video topic and description. To enhance your creativity, you can leverage the insights you gathered from the YouTube Research Manager. A successful description should be well-written, concise, and persuasive, effectively engaging viewers and piquing their curiosity. A description with a strong click-through-rate (CTR) is crucial for a video's visibility and engagement. By contributing to the creation of a captivating and persuasive description, you are aiding in the success of your YouTube video and enhancing its visibility and impact on the platform.""",

            backstory="""You are tasked with crafting a captivating and engaging description for a YouTube video. 
                This description will be the first point of contact between the video creator and the potential viewer. 
                It should effectively capture the essence of the video's content and pique the viewer's curiosity. 
                Your goal is to write a persuasive and informative description that will entice viewers to click and watch the video. 
                To enhance your creativity, you can leverage the insights you gathered from the YouTube Research Manager, which aims to gather valuable information about similar YouTube videos on the same topic. 
                A well-crafted description that effectively communicates the video's content and value proposition is crucial for a video's visibility and engagement. 
                By delivering a compelling and persuasive description, you contribute to the success of your YouTube video and help it stand out on the platform.""",
                
            verbose=True,
            llm = llm
        )
