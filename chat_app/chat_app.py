from rxconfig import config
import reflex as rx
from chat_app import style
from chat_app.state import State
filename = f"{config.app_name}/{config.app_name}.py"


def qa(question: str, answer: str) -> rx.Component:
    return rx.box(
        rx.box(
            rx.text(question, text_align="right"),
            style=style.question_style,
        ),
        rx.box(
            rx.text(answer, text_align="left"),
            style=style.answer_style,
        ),
        margin_y="1em",
    )



# def chat() -> rx.Component:
#     return rx.box(
#         rx.foreach(
#             State.chat_history,
#             lambda messages: qa(messages[0], messages[1]),
#         )
#     )



def chat() -> rx.Component:
    """List all the messages in a single conversation."""

    chatbox = rx.box(
        rx.foreach(State.chat_history,
                   lambda messages: qa(messages[0], messages[1]),),
                   width="100%")

    return rx.vstack(
        chatbox,
        py="8",
        flex="1",
        width="100%",
        max_width="60em",
        padding_x="4px",
        align_self="center",
        overflow="hidden",
        padding_bottom="5em",
    )



def action_bar() -> rx.Component:
    return rx.center( 
            rx.hstack(
                 rx.input(
                    value=State.question,
                    placeholder="Ask a question about USF",
                    on_change=State.set_question,
                    # size=1,
                    style=style.input_style,
                ),
                rx.button(
                    "Send",
                    on_click=State.answer,
                    color_scheme="teal",
                ),
                rx.badge(State.rag_type,
                         rx.tooltip(rx.icon("info", size=14), 
                        content="The current selected rag type."),
                        variant="soft"),
            
            align_items="stretch",
            # width="100%",
                
                ),

        position="sticky",
        bottom="0",
        left="0",
        padding_y="16px",
        backdrop_filter="auto",
        backdrop_blur="lg",
        border_top=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        align_items="stretch",
        width="100%",
    )



def navbar():

    dropdown_menu = rx.hstack(
        rx.badge(State.rag_type,
                 rx.tooltip(rx.icon("info", size=14), 
                content="The current selected rag type."),
                 variant="soft"),
        rx.select(State.rag_values,
            value=State.current_rag_val,
            on_change=State.set_current_rag_val
        ),
        rx.button(
            "Set",
            type="submit",
            on_click=State.change_rag_type,
        ),
            # justify_content="space-between",
        align_items="stretch",

    )

    openai_key_input = rx.input(
            value=State.openai_api_key,
            placeholder="Enter your openai key",
            on_change=State.set_openai_api_key,
            style=style.input_style,
            type="password",
    )

    settings_box = rx.vstack(
        openai_key_input,
        dropdown_menu,
    )


    logo_box = rx.hstack(
                rx.avatar(fallback="BB", variant="solid"),
                rx.heading("Bull Buddy"),
                align_items="center",
            )


    return rx.box(
        rx.hstack(logo_box,
            settings_box,
            justify_content="space-between",
            align_items="center",
        ),

        backdrop_filter="auto",
        backdrop_blur="lg",
        padding="12px",
        border_bottom=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        position="sticky",
        top="0",
        z_index="100",
        align_items="center",

    )



def index() -> rx.Component:
    """The main app."""
    return rx.chakra.vstack(
        navbar(),
        chat(),
        action_bar(),
        background_color=rx.color("mauve", 1),
        color=rx.color("mauve", 12),
        min_height="100vh",
        align_items="stretch",
        spacing="0",
    )


app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="teal",
    ),
)

app.add_page(index)