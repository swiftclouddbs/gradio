import gradio as gr

def search_in_file(filename, search_term):
    """
    Search for a term in a file and return lines containing that term.
    
    Args:
        filename (str): The path to the file to search.
        search_term (str): The term to search for in the file.
        
    Returns:
        str: Lines containing the search term.
    """
    results = []
    try:
        with open(filename, 'r') as text_file:
            lines = text_file.readlines()
            for line in lines:
                if search_term in line:
                    results.append(line.strip())  # Use strip to remove extra new lines
    except FileNotFoundError:
        return f"Error: The file '{filename}' was not found."
    except Exception as e:
        return f"An error occurred: {e}"
    
    return "\n".join(results) if results else "No matches found."

# Define the Gradio interface
def gradio_interface(search_term):
    return search_in_file("holiday_list.txt", search_term)

# Set up the Gradio app
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Search Term"),
    outputs=gr.Textbox(label="Results"),
    title="Holidays!",
    description="Enter a holiday and check to see if the office is closed."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
