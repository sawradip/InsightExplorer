import base64
from io import BytesIO
from typing import Optional, Dict, List
import gradio as gr
from PIL import Image
from pandas import DataFrame
from lida import Manager
from lida.utils import read_dataframe
from lida.datamodel import TextGenerationConfig, Goal
from dotenv import load_dotenv

load_dotenv(".env")

lida_manager = Manager()


MAX_EXPLANATIONS = 10
MAX_VISUALIZATIONS = 1
SUMMARY_LLM_MODEL = "gpt-3.5-turbo-16k"
TASK_LLM_MODEL = "gpt-4o"

NON_MINIMAL_MODE = True


def display_dataframe(data_filepath: str, row_count: int = 5):
    """
    Display a dataframe and provide options to select columns for analysis.

    Args:
        data_filepath (str): The file path to the data file.
        row_count (int, optional): The number of rows to display.
        Defaults to 5.

    Returns:
        tuple: A tuple containing the dataframe viewer and the
            selected columns checkbox group.
    """
    df = read_dataframe(data_filepath)
    df_viewer = gr.DataFrame(data_filepath, row_count=row_count, visible=True)
    selected_columns = gr.CheckboxGroup(
        df.columns.tolist(),
        value=df.columns.tolist(),
        visible=True,
        label="Select Columns",
        info="The ones you want to analyze."
    )
    return df_viewer, selected_columns


def modify_dataframe(
    data_filepath: str,
    col_list: list
) -> 'DataFrame':
    """
    Modify a dataframe by selecting specific columns.

    Args:
        data_file (str): The file path to the data file.
        col_list (list): List of column names to select from the dataframe.

    Returns:
        DataFrame: A dataframe containing only the selected columns.
    """
    df = read_dataframe(data_filepath)
    df_viewer = df[col_list]
    return df_viewer


def get_summary(
    data_filepath: str,
    textgen_config: Optional[TextGenerationConfig] = None
) -> Dict:
    """
    Generate a summary for the given data file using the LIDA manager.

    Args:
        data_filepath (str): The file path to the data file.
        textgen_config (Optional[TextGenerationConfig]): Configuration for text generation.
            If None, defaults to a pre-defined configuration.

    Returns:
        Dict: A dictionary containing the summary of the data.
    """
    global lida_manager
    data_summary = lida_manager.summarize(
        data=data_filepath,
        file_name="",
        n_samples=3,
        textgen_config=textgen_config or TextGenerationConfig(
            model=SUMMARY_LLM_MODEL
            ),
        summary_method="llm",
    )

    return data_summary


def display_summary(data_filepath: str) -> List[gr.Markdown]:
    """
    Display a summary of the data file in a markdown format.

    This function retrieves a summary of the data file using the global LIDA manager
    and formats it into a markdown string that can be displayed in a GUI.

    Args:
        data_filepath (str): The file path to the data file.

    Returns:
        List[gr.Markdown]: A list containing markdown components for the name,
            file, dataset description, and detailed field information.
    """
    global lida_manager

    data_summary = get_summary(data_filepath, textgen_config=None)

    # print(data_summary)
    name = data_summary["name"]
    filename = data_summary["file_name"]
    data_description = data_summary["dataset_description"]
    fields = data_summary["fields"]

    data_info_str = f"## {name}\n"
    data_info_str += f"### File: {filename}\n"
    data_info_str += f"### {data_description}\n"

    data_info = gr.Markdown(data_info_str)

    # column_descriptors = []
    summary_full_str = ""

    for field in fields:
        field_name = field['column']
        field_properties = field['properties']
        field_dtype = field_properties.pop("dtype")
        field_description = field_properties.pop("description")

        field_header = (
            f'<br><h3> {field_name} ({field_dtype}) - '
            f'{field_description}</h3>'
        )

        field_details = []
        for detail_name, detail_value in field_properties.items():
            field_details.append(
                f"> {detail_name}: "
                f"{', '.join(map(str, detail_value)) if isinstance(detail_value, list) else str(detail_value)}"
                )

        field_details_str = "<br>".join(field_details)

        summary_full_str += field_header
        summary_full_str += field_details_str

    summary_descriptor = gr.Markdown(summary_full_str, visible=True)

    return [data_info, summary_descriptor]


def base64_to_image(base64_string: str) -> Image:
    """
    Convert a base64 encoded string to an image.

    Args:
        base64_string (str): The base64 encoded string representing the image.

    Returns:
        Image: An image object that can be manipulated or displayed in Python.
    """
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


def display_goals(
    data_filepath: str,
    n_goals: int,
    insight_persona: str
) -> List[gr.Accordion]:
    """
    Display the goals based on the data file path, number of goals, and insight persona.

    This function retrieves a summary of the data from the given file path and uses the LIDA manager
    to generate goals based on the insight persona. It then creates UI components to display these goals.

    Args:
        data_filepath (str): The file path to the data for which goals are to be generated.
        n_goals (int): The number of goals to generate.
        insight_persona (str): The persona to use for insight generation.

    Returns:
        List[gr.Accordion]: A list of gr.Accordion objects that represent the goal cards to be displayed in the UI.
    """
    global lida_manager

    data_summary = get_summary(data_filepath, textgen_config=None)

    goals = lida_manager.goals(data_summary, n=n_goals, persona=insight_persona)
    goal_cards = []
    for goal in goals:
        with gr.Accordion(f"{goal.question}", open=False, visible=True) as goal_card:
            question_holder = gr.Text(goal.question)
            visualization_holder = gr.Text(goal.visualization)
            rationale_holder = gr.Text(goal.rationale)
            visualization_text = gr.Markdown(f"""*<span style="color: grey;"><b>Visualization:<b>{goal.visualization}</span>*""")
            with gr.Row(equal_height=True):   
                with gr.Column(scale=3):
                    rationale_text = gr.Markdown(f"**Rationale:** {goal.rationale}")
                generate_button = gr.Button("Visualize", scale=1, visible=True)

        goal_card_components = [goal_card,
                                question_holder,
                                visualization_holder,
                                rationale_holder,
                                visualization_text,
                                rationale_text,
                                generate_button,
                                # *visualization_blocks
                                ]
        goal_cards.extend(goal_card_components)

    for i in range(MAX_GOALS - len(goals)):
        with gr.Accordion("", open=False, visible=False) as goal_card:
            question_holder = gr.Textbox("", visible=False)
            visualization_holder = gr.Textbox("", visible=False)
            rationale_holder = gr.Textbox("", visible=False)
            visualization_text = gr.Markdown("",)
            with gr.Row(equal_height=True): 
                with gr.Column(scale=3):
                    rationale_text = gr.Markdown("")
                generate_button = gr.Button(scale=1, visible=False)

        goal_card_components = [goal_card, 
                                question_holder, 
                                visualization_holder, 
                                rationale_holder, 
                                visualization_text, 
                                rationale_text, 
                                generate_button, 
                                # *visualization_blocks
                                ]
        goal_cards.extend(goal_card_components)
    return goal_cards

def display_visualization(data_filepath, goal_question, goal_visualization, goal_rationale):
    print("display_visualization clicked")
    global lida_manager

    data_summary = get_summary(data_filepath, textgen_config=None)
    
    goal = Goal(question = goal_question,
                visualization = goal_visualization,
                rationale = goal_rationale)

    charts = lida_manager.visualize(summary=data_summary, goal=goal, library="matplotlib")

    visualization_pieces = []
    for chart in charts:
        # if chart.status and not chart.error:
        # with gr.Row(equal_height=True): 
        #     chart_img = gr.Image(base64_to_image(chart.raster), visible=True) 
        #     code_block = gr.Code(chart.code, language="python", visible=True)

        # visualization_pieces.append(chart_img)
        # visualization_pieces.append(code_block)

        # with gr.Row(equal_height=True): 
        chart_img = gr.Image(base64_to_image(chart.raster), visible=True)
        with gr.Accordion("Code Here", visible=NON_MINIMAL_MODE) as code_holder:
            code_block = gr.Code(chart.code, language="python", visible=True)
        visualization_pieces.append(chart_img)
        visualization_pieces.append(code_holder)
        visualization_pieces.append(code_block)


    for i in range(MAX_VISUALIZATIONS - len(charts)):
        # with gr.Row(equal_height=True): 
        #     chart_img = gr.Image(visible=False) 
        #     code_block = gr.Code(visible=False)
            
        # visualization_pieces.append(chart_img)
        # visualization_pieces.append(code_block)
    
        chart_img = gr.Image(visible=False)
        with gr.Accordion("Code Here", visible=False) as code_holder:
            code_block = gr.Code(language="python", visible=False)
        visualization_pieces.append(chart_img)
        visualization_pieces.append(code_holder)
        visualization_pieces.append(code_block)

    with gr.Row():
        modification_text = gr.Textbox("", label="Write expected modifications(seperated by commma)", interactive=True, visible=True, scale=3)
        modify_button = gr.Button("Modify Visualizations", scale=1, visible=True)
    recommendation_button = gr.Button("Show Similar Visualizations", scale=0.5, visible=True)

    return visualization_pieces + [modification_text, modify_button, recommendation_button]
            

# instructions = ["convert this to a bar chart", "change the theme to red", "translate the title to Bangla"]
# edited_charts = lida.edit(code=charts[0].code,  summary=summary, instructions=instructions, library='seaborn', 
#                         # textgen_config=textgen_config
#                         )

def display_modified_visualization(data_filepath, existing_code, modification_instructions):
    print("display_modified_visualization clicked")
    global lida_manager

    data_summary = get_summary(data_filepath, textgen_config=None)

    modification_instruction_list = modification_instructions.split(",")

    edited_charts = lida_manager.edit(code=existing_code,  
                                summary=data_summary, 
                                instructions=modification_instruction_list, 
                                library='seaborn', 
                                textgen_config=TextGenerationConfig(model=TASK_LLM_MODEL),
                                )
    print(edited_charts)
    visualization_pieces = []
    for chart in edited_charts:
        # if chart.status and not chart.error:
        # with gr.Row(equal_height=True): 
            # chart_img = gr.Image(base64_to_image(chart.raster), visible=True) 
            # code_block = gr.Code(chart.code, language="python", visible=True)
        # visualization_pieces.append(chart_img)
        # visualization_pieces.append(code_block)

        chart_img = gr.Image(base64_to_image(chart.raster), visible=True)
        with gr.Accordion("Code Here", visible=NON_MINIMAL_MODE) as code_holder:
            code_block = gr.Code(chart.code, language="python", visible=True)
        visualization_pieces.append(chart_img)
        visualization_pieces.append(code_holder)
        visualization_pieces.append(code_block)



    for i in range(MAX_VISUALIZATIONS - len(edited_charts)):
        # with gr.Row(equal_height=True): 
        #     chart_img = gr.Image(visible=False) 
        #     code_block = gr.Code(visible=False)
            
        # visualization_pieces.append(chart_img)
        # visualization_pieces.append(code_block)
        chart_img = gr.Image(visible=False)
        with gr.Accordion("Code Here", visible=False) as code_holder:
            code_block = gr.Code(language="python", visible=False)
        visualization_pieces.append(chart_img)
        visualization_pieces.append(code_holder)
        visualization_pieces.append(code_block)
        
    print("display_modified_visualization exited")
    return visualization_pieces



def display_recommendation_visualization(data_filepath, existing_code, n = 5):
    print("display_recommendation_visualization clicked")

    global lida_manager

    data_summary = get_summary(data_filepath, textgen_config=None)

    recommendations = lida_manager.recommend(existing_code, 
                                    summary=data_summary, 
                                    n=n,  
                                    textgen_config=TextGenerationConfig(model=TASK_LLM_MODEL),
                                    )

    recommendation_gallery = gr.Gallery(
                                    [base64_to_image(recommendation.raster) for recommendation in recommendations],
                                    visible=True, 
                                    preview=True,
                                    # allow_preview = False            
                                        )

    return recommendation_gallery


def display_visualization_only_question(data_filepath, goal_question):
    print("visualization_only_question clicked")
    global lida_manager

    data_summary = get_summary(data_filepath, textgen_config=None)

    # goal = Goal(question = goal_question,
    #         visualization = "",
    #         rationale = "")
    charts = lida_manager.visualize(
        summary=data_summary,
        goal=goal_question,
        textgen_config=TextGenerationConfig(model=TASK_LLM_MODEL),
        library="seaborn")

    visualization_pieces = []
    for chart in charts:
        chart_img = gr.Image(base64_to_image(chart.raster), visible=True)
        with gr.Accordion("Code Here", visible=NON_MINIMAL_MODE) as code_holder:
            code_block = gr.Code(chart.code, language="python", visible=True)
        visualization_pieces.append(chart_img)
        visualization_pieces.append(code_holder)
        visualization_pieces.append(code_block)

    for i in range(MAX_VISUALIZATIONS - len(charts)):
        # with gr.Row(equal_height=True): 
        #     chart_img = gr.Image(visible=False, scale=2) 
        #     code_block = gr.Code(visible=False, scale=1) 
            
        # visualization_pieces.append(chart_img)
        # visualization_pieces.append(code_block)
        chart_img = gr.Image(visible=False)
        with gr.Accordion("Code Here", visible=False) as code_holder:
            code_block = gr.Code(language="python", visible=False)
        visualization_pieces.append(chart_img)
        visualization_pieces.append(code_holder)
        visualization_pieces.append(code_block)

    with gr.Row():
        modification_text = gr.Textbox("", label="Write expected modifications(seperated by commma)", interactive=True, visible=True, scale=3)
        modify_button = gr.Button("Modify Visualizations", scale=1, visible=True)
    
    with gr.Row():
            explanation_button = gr.Button(visible=True)
            evaluation_button = gr.Button(visible=True)
            recommendation_button = gr.Button(visible=True)


    print("visualization_only_question exiting")
    return visualization_pieces + [ modification_text, 
                                    modify_button, 
                                    explanation_button, 
                                    evaluation_button,
                                    recommendation_button,
                                    ]

def display_explanation(data_filepath, existing_code):
    print("visualization explanation clicked")
    global lida_manager

    print(existing_code)
    data_summary = get_summary(data_filepath, textgen_config=None)
    
    print("### Explanation Code Sending:", existing_code)
    explanation_items = lida_manager.explain(code=existing_code)[0]
    print("### Explanation Recieved:", explanation_items)

    explanation_evaluation_blocks = []
    for i, explanation_item in enumerate(explanation_items):
        section = explanation_item["section"]
        code = explanation_item["code"]
        explanation = explanation_item["explanation"]
        print("#####", i, code)
        with gr.Accordion(f"Code Block {i+1}", visible=True) as explanator_evaluator:
            code_block = gr.Code(code, visible=True)
            small_text1 = gr.Markdown(f"<b>Section</b>: {section}")
            small_text2 = gr.Markdown(f"<b>Explanation</b>: {explanation}")

        explanation_evaluation_blocks.append(explanator_evaluator)
        explanation_evaluation_blocks.append(code_block)
        explanation_evaluation_blocks.append(small_text1)
        explanation_evaluation_blocks.append(small_text2)
        
    for _ in range(MAX_EXPLANATIONS - len(explanation_items)):
        with gr.Accordion("", visible=False) as explanator_evaluator:
            code_block = gr.Code(visible=False)
            small_text1 = gr.Markdown("")
            small_text2 = gr.Markdown("")

        explanation_evaluation_blocks.append(explanator_evaluator)
        explanation_evaluation_blocks.append(code_block)
        explanation_evaluation_blocks.append(small_text1)
        explanation_evaluation_blocks.append(small_text2)

    return explanation_evaluation_blocks


def display_evaluation(data_filepath, existing_code,goal_question):
    print("visualization explanation clicked")
    global lida_manager

    data_summary = get_summary(data_filepath, textgen_config=None)

    goal = Goal(question = goal_question,
                visualization = goal_question,
                rationale = "")
    explanation_items = lida_manager.evaluate(code=existing_code, goal = goal)[0]

    explanation_evaluation_blocks = []
    for i, explanation_item in enumerate(explanation_items):
        rationale = explanation_item["rationale"]
        dimension = explanation_item["dimension"]
        score = explanation_item["score"]
        with gr.Accordion(rationale, visible=True) as explanator_evaluator:
            code_block = gr.Code(visible=False)
            small_text1 = gr.Markdown(f"<b>Score</b>: {score}")
            small_text2 = gr.Markdown(f"<b>Dimension</b>: {dimension}")

        explanation_evaluation_blocks.append(explanator_evaluator)
        explanation_evaluation_blocks.append(code_block)
        explanation_evaluation_blocks.append(small_text1)
        explanation_evaluation_blocks.append(small_text2)
        
    for _ in range(MAX_EXPLANATIONS - len(explanation_items)):
        with gr.Accordion("", visible=False) as explanator_evaluator:
            code_block = gr.Code(visible=False)
            small_text1 = gr.Markdown("")
            small_text2 = gr.Markdown("")

        explanation_evaluation_blocks.append(explanator_evaluator)
        explanation_evaluation_blocks.append(code_block)
        explanation_evaluation_blocks.append(small_text1)
        explanation_evaluation_blocks.append(small_text2)

    return explanation_evaluation_blocks



import gradio as gr

title_html = (
    """<br><h1 style="text-align:center; font-size: 50px; """
    """text-shadow: -1px 0 white, 0 1px white, 1px 0 white, 0 -1px white;"> """
    """<span style="color:#000000;">Insight</span>"""
    """<span style="color:#6bc7f0;">Explorer</span> </h1>"""
)
subtitle_html = (
    """<h2 style="text-align:center; """
    """text-shadow: -1px 0 white, 0 1px white, 1px 0 white, 0 -1px white;"> """
    """<span style="color:#000000;">Talk to your data files — """
    """Get Insights</span></h2>"""
)

with gr.Blocks(
    css="footer{display:none !important}"
) as demo:
    gr.HTML(title_html)
    gr.HTML(subtitle_html)
    # name = gr.Textbox(label="OpenAI API Key", type= "password")
    data_file = gr.File(height="0.5in")
    selected_col_list = gr.CheckboxGroup(
        visible=False,
        label="Select Columns",
        info="The ones you want to analyze."
        )
    data_viewer = gr.DataFrame(
        visible=False,
        label="Loaded Data",
        show_label=True,
        interactive=False)
    
    data_file.upload(
        display_dataframe,
        inputs=[data_file],
        outputs=[data_viewer, selected_col_list]
        )
    selected_col_list.change(
        modify_dataframe,
        inputs=[data_file, selected_col_list],
        outputs=[data_viewer]

    )
    with gr.Tab("Summary"):
        show_summary_btn = gr.Button("Show Summary", interactive=True)
        data_info = gr.Markdown()
        summary_descriptor = gr.Markdown()

        show_summary_btn.click( display_summary,
                            inputs=[data_viewer],
                            outputs=[
                                data_info,
                                summary_descriptor
                                ]
                            )
    with gr.Tab("Insights"):
        with gr.Row():
            persona_txt = gr.Text("Business Analyst at at large Corporate Organization.", interactive=True, scale=3)
            goal_counter = gr.Slider(minimum=2, maximum=10, step=1, interactive=True, scale=1)
        
        propose_insights_btn = gr.Button("Propose Insights")


        goal_cards = []
        for i in range(MAX_GOALS):
            with gr.Accordion("", visible=NON_MINIMAL_MODE) as goal_card:
                question_holder = gr.Textbox("", visible=False)
                visualization_holder = gr.Textbox("", visible=False)
                rationale_holder = gr.Textbox("", visible=False)
                visualization_text = gr.Markdown("",)
                with gr.Row(equal_height=True):   
                    with gr.Column(scale=3):
                        rationale_text = gr.Markdown("")
                    visualize_button = gr.Button(scale=1, visible=False)

                visualization_blocks = []
                for n_row in range(MAX_VISUALIZATIONS):
                    # with gr.Row(equal_height=True): 
                    chart_img = gr.Image(visible=False, scale=2)
                    with gr.Accordion("", visible=NON_MINIMAL_MODE) as code_holder:
                        code_block = gr.Code(visible=False, scale=1) 
                    visualization_blocks.append(chart_img)
                    visualization_blocks.append(code_holder)
                    visualization_blocks.append(code_block)
                with gr.Row():
                    modification_text = gr.Textbox("", visible=False, scale=3)
                    modify_button = gr.Button(scale=1, visible=False)
                recommendation_button = gr.Button(scale=1, visible=False)

                recommendation_gallery = gr.Gallery(visible=False)

            visualize_button.click(
                        display_visualization,
                        inputs = [data_file, question_holder, visualization_holder, rationale_holder],
                        outputs = visualization_blocks + [modification_text, modify_button, recommendation_button]
            )
            modify_button.click(
                        display_modified_visualization,
                        inputs = [data_file, code_block, modification_text],
                        outputs = visualization_blocks
            )

            recommendation_button.click(
                display_recommendation_visualization,
                inputs = [data_file, code_block],
                outputs = [recommendation_gallery]
            )

            goal_card_components = [goal_card, 
                                    question_holder, 
                                    visualization_holder, 
                                    rationale_holder, 
                                    visualization_text, 
                                    rationale_text, 
                                    visualize_button, 
                                    # *visualization_blocks,
                                    ]
            goal_cards.extend(goal_card_components)

        
    with gr.Tab("Queries"):
        with gr.Row():
            query_txt = gr.Text(interactive=True, scale=3)
            query_btn = gr.Button("Ask or Query")

        visualization_blocks = []
        for n_row in range(MAX_VISUALIZATIONS):
            chart_img = gr.Image(visible=False, scale=2)
            with gr.Accordion("Code Here", visible=NON_MINIMAL_MODE) as code_holder:
                code_block = gr.Code(visible=False, scale=1) 
            visualization_blocks.append(chart_img)
            visualization_blocks.append(code_holder)
            visualization_blocks.append(code_block)

        with gr.Row():
            modification_text = gr.Textbox("", visible=False, scale=3)
            modify_button = gr.Button(scale=1, visible=False)

        with gr.Row():
            recommendation_button = gr.Button("Similar Plots", scale=3, visible=False)
            evaluation_button = gr.Button("Analyse Answer", scale=1, visible=False)
            explanation_button = gr.Button("Explain Code", scale=1, visible=False)

        recommendation_gallery = gr.Gallery(visible=False)

        explanation_evaluation_blocks = []
        for _ in range(MAX_EXPLANATIONS):
            with gr.Accordion("", visible=False) as explanator_evaluator:
                code_subblock = gr.Code(visible=False)
                small_text1 = gr.Markdown("")
                small_text2 = gr.Markdown("")

            explanation_evaluation_blocks.append(explanator_evaluator)
            explanation_evaluation_blocks.append(code_subblock)
            explanation_evaluation_blocks.append(small_text1)
            explanation_evaluation_blocks.append(small_text2)

        query_btn.click(
            display_visualization_only_question,
            inputs=[data_file, query_txt],
            outputs=visualization_blocks + [
                modification_text,
                modify_button,
                explanation_button,
                evaluation_button,
                recommendation_button
                ]
        )

        modify_button.click(
            display_modified_visualization,
            inputs=[data_file, code_block, modification_text],
            outputs=visualization_blocks
        )

        recommendation_button.click(
            display_recommendation_visualization,
            inputs=[data_file, code_block],
            outputs=[recommendation_gallery]
        )

        explanation_button.click(
            display_explanation,
            inputs=[data_file, code_block],
            outputs=explanation_evaluation_blocks
        )

        evaluation_button.click(
            display_evaluation,
            inputs=[data_file, code_block, query_txt],
            outputs=explanation_evaluation_blocks
        )
                     
    propose_insights_btn.click(
        display_goals,
        inputs=[
            data_viewer,
            goal_counter,
            persona_txt],
        outputs=goal_cards
        )


demo.launch(
    share=False,
    debug=True,
    server_name="0.0.0.0",
    server_port=7232
    )









