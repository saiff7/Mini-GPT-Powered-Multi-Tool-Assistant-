from llama_cpp import Llama
import json
import requests
from datetime import datetime
import os
import shutil
import PIL
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # Use non-interactive backend for matplotlib

# Enable/disable GPU acceleration (requires CUDA support)
USE_GPU = False

# Verbosity level (0 = only final assistant messages, 1 = function calls, 2 = function calls and their results)
VERBOSITY = 2

# Enable the display of matplotlib plots once created
SHOW_PLOTS = True 

# Model path relative to script location
# Download from https://huggingface.co/katanemo/Arch-Function-7B.gguf/resolve/main/Arch-Function-7B-Q6_K.gguf?download=true
# or            https://huggingface.co/mradermacher/Arch-Function-7B-i1-GGUF/resolve/main/Arch-Function-7B.i1-Q4_K_S.gguf
# MODEL_PATH = "models/Arch-Function-7B-Q6_K.gguf"      # 6-bit quantized version of the model; slower but more accurate
MODEL_PATH = "models/Arch-Function-7B.i1-Q4_K_S.gguf"   # 4-bit quantized version of the model; faster but less accurate (recommended for CPU-only usage)

# Model parameters
TEMPERATURE = 0.0   # Need deterministic results for function calls
MAX_TOKENS = 768
CONTEXT_LENGTH = 4096
BATCH_SIZE = 512

# Color codes for console output
color_dict = {'red': 31, 'green': 32, 'yellow': 33, 'blue': 34, 'magenta': 35, 'white': 37}
current_color = "white"
current_column = 0      # Current column in the console output

# OpenWeatherMap API key - replace with your own that you can get for free from https://openweathermap.org/api
# General advice: Do not hardcode API keys in your code. Use environment variables or secure storage.
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

TASK_PROMPT = "You are a helpful assistant."

TOOL_PROMPT = """
You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_text}
</tools>
""".strip()

FORMAT_PROMPT = """
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
""".strip()

# Define available tools, i.e., functions that the model can call, in OpenAI format (most common format)
get_weather = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "str",
                    "description": "The city, state, and/or country, e.g., San Francisco, New York",
                },
                "unit": {
                    "type": "str",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature to return",
                },
            },
            "required": ["location"],
        },
    },
}

create_bar_chart = {
    "type": "function",
    "function": {
        "name": "create_bar_chart",
        "description": "Create a chart showing vertical bars whose height (y-value) is exactly identical to the value of a given variable measured across multiple \
            categories. Provide the category names and their corresponding y-values (bar heights) in separate arrays, in the order to be displayed. \
            Give a chart title and and a label for the y-axis that clearly indicate what is being shown.",
        "parameters": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {"type": "str"},
                    "description": "Category labels for the vertical bars (from left to right) shown in the chart",
                },
                "y_values": {
                    "type": "array",
                    "items": {"type": "float"},
                    "description": "Exact bar height (y-axis value) as obtained for each category, precisely replicated in the same order as the categories array",
                },
                "title": {
                    "type": "str",
                    "description": "The title of the chart",
                },
                "y_axis_label": {
                    "type": "str",
                    "description": "Label explaining the values on the y-axis (bar heights).",
                }
            },
            "required": ["categories", "y_values", "title", "y_axis_label"],
        },
    },
}

# List of available functions in OpenAI format (most common format)
function_list = [get_weather, create_bar_chart]

def set_color(color_string):
    global current_color
    current_color = color_string

def convert_tools(tools):
    return "\n".join([json.dumps(tool) for tool in tools])

# Helper function to create the system prompt for our model
def format_system_prompt(tools):
    tool_text = convert_tools(tools)
    return (
        TASK_PROMPT
        + "\n\n"
        + TOOL_PROMPT.format(tool_text=tool_text)
        + "\n\n"
        + FORMAT_PROMPT
        + "\n"
    )

def get_weather(location, unit="celsius"):
    """Fetch current weather data for a given location."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"]
            }
        else:
            return {"error": f"Could not fetch weather data: {data.get('message', 'Unknown error')}"}
    except Exception as e:
        return {"error": f"Error fetching weather data: {str(e)}"}

def create_bar_chart(categories, y_values, title, y_axis_label):
    """Create,display and save a plot with the given data."""
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(categories, y_values, color='skyblue', edgecolor='black')
        plt.title(title, fontsize=14, pad=20)
        plt.ylabel(y_axis_label, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"plot_{timestamp}.png"
        plt.savefig(filename)
        
        if SHOW_PLOTS:
            PIL.Image.open(filename).show()
        
        return {"filename": filename}
    except Exception as e:
        return {"error": f"Error creating plot: {str(e)}"}

# Load the model
print(f"Loading model from {MODEL_PATH}...")
print(f"GPU acceleration: {'enabled' if USE_GPU else 'disabled'}")

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=35 if USE_GPU else 0,
    n_ctx=CONTEXT_LENGTH,
    n_batch=BATCH_SIZE,
    flash_attn=True,
    verbose=False
)

def get_terminal_width():
    size = shutil.get_terminal_size()
    return size.columns

def print_chunk(text, end="\n", flush=False):
    """Helper function for print_colored."""
    
    if (VERBOSITY == 0) and not (current_color in ["white", "green"]):  # Only allow user and final assoistant messages  
        return
    if (VERBOSITY == 1) and (current_color == "magenta"):   # Do not print results of function calls
        return
        
    print(f"\033[{color_dict[current_color]}m{text}\033[0m", end=end, flush=flush)

def print_colored(text, end="\n", flush=False):
    """Print colored text to the console if verbosity level allows it. Make sure that new lines are handled correctly."""
    global current_column
    
    chunks = text.split(" ")
    
    while chunks:
        chunk = chunks.pop(0)
        if chunk == "":    # Chunk starts or ends on a space
            if (chunks):   # Chunk starts on a space
                if current_column >= get_terminal_width() - len(chunks[0]) - 10: # New line if current column is close to the terminal width
                    print()
                    current_column = 0
                    continue
                    
            elif current_column >= get_terminal_width() - 10:   # Chunk ends on a space
                print()
                current_column = 0
                return
            
        elif current_column + len(chunk) >= get_terminal_width() - 4:
            print()
            current_column = 0

        if chunks:
            chunk += " "
            print_chunk(chunk, end="")
        else:
            print_chunk(chunk, end=end, flush=flush)
           
        for char in chunk:
            if char == "\n":
                current_column = 0
            else:
                current_column += 1
    
    if end == "\n":
        current_column = 0
        
def init_chat_messages():
    """Initialize chat messages with system message."""
    return [{"role": "system", "content": format_system_prompt(function_list)}]

def handle_function_call(function_call):
    """Handle function calls from the LLM."""
    function_name = function_call.get("name")
    arguments = function_call.get("arguments", "{}")
    set_color("blue")
    print_colored(f"\nCalling function: {function_name} with arguments: {arguments}\n")    
    if function_name == "get_weather":
        return get_weather(**arguments)
    elif function_name == "create_bar_chart":
        return create_bar_chart(**arguments)
    else:
        return {"error": f"Unknown function: {function_name}"}

def get_streaming_response(messages):
    """Generate streaming response from the model and extract list of function calls."""
    response = ""
    current_function_call = ""
    function_calls = []
    parsing_function_call = False
    
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True
    )
    
    for chunk in stream:
        if chunk and "choices" in chunk:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                content = delta["content"] 
            
                # Handle function calls
                if "<tool_call>" in content:
                    parsing_function_call = True
                    set_color("yellow")
                elif "</tool_call>" in content:
                    function_calls.append(json.loads(current_function_call.strip()))               
                    current_function_call = ""
                    parsing_function_call = False
                
                print_colored(content, end="", flush=True)
                response += content
        
                if parsing_function_call:
                    if not ("<tool_call>" in content):
                        current_function_call += content    
                else:
                    set_color("green")    
                    
    print()  # New line after response
    return response, function_calls

def main():
    messages = init_chat_messages()
    function_results = ""
    global current_column
    
    print_colored("\nChat session started. Type 'new' for a new chat, 'end' to exit.")
    print_colored("\nExample requests:\n")
    print_colored("- What's the weather in Boston?")
    print_colored("- Which city has the more pleasant weather right now, Boston or Rome?")
    print_colored("- Create a chart showing the exact current temperatures in degrees Celsius in the following cities: Boston, Stockholm, Nairobi, and Sydney, in ascending order of temperatures.'")
    print_colored("- Create a bar chart showing the value of the first 20 prime numbers in ascending order.")
    print_colored("- Pick 5 major US cities and obtain their current wind speed. Sort the cities in ascending order of their wind speed while making sure that the associations between cities and " + 
          "their wind speeds remain intact. Then create a bar chart illustrating the sorted wind speed vs. city data.")
    
    while True:
        if not function_results:
            set_color("white")
            user_input = input("\nYou: ").strip()
            current_column = 0
                        
            if user_input.lower() == "end":
                break
            elif user_input.lower() == "new":
                messages = init_chat_messages()
                print("\nStarting new chat session...")
                continue
            elif not user_input:
                continue
                
            # Add user message to chat
            messages.append({"role": "user", "content": user_input})
            
        # Get and print assistant's response
        set_color("green")
        print_colored("\nAssistant: ", end="", flush=True)
        response, function_calls = get_streaming_response(messages)
        
        # Add assistant's response to chat history
        messages.append({"role": "assistant", "content": response})
        function_results = ""
        
        if function_calls:        
            for call in function_calls:
                execution_result = handle_function_call(call)
                
                if "error" in execution_result:
                    set_color("red")
                else:   
                    set_color("magenta")
                
                formatted_response = f"<tool_response>\n{json.dumps(execution_result)}\n</tool_response>" 
                print_colored(formatted_response, end="\n", flush=True)
                function_results += formatted_response
            
            messages.append({"role": "user", "content": function_results})
        
if __name__ == "__main__":
    main()
