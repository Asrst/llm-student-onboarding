# style.py
import reflex as rx

# Common styles for questions and answers.
shadow = "rgba(0, 0, 0, 0.15) 0px 2px 8px"
chat_margin = "20%"
message_style = dict(
    padding="1em",
    border_radius="5px",
    margin_y="0.5em",
    box_shadow=shadow,
)

# Set specific styles for questions and answers.
question_style = message_style | dict(
    bg=rx.color("mauve", 4), margin_left=chat_margin
)

answer_style = message_style | dict(
    bg=rx.color("accent", 4), margin_right=chat_margin
)

# Styles for the action bar.
input_style = dict(
    border_width="1px", padding="1em", box_shadow=shadow, 
    align_items="stretch",
    width="250px"
)

button_style = dict(bg="#CEFFEE", box_shadow=shadow)

# openai_input_style = {
#     "color": "white",
#     "margin-top": "2rem",
#     "margin-bottom": "1rem",
# }

