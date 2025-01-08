import gradio as gr
import pandas as pd
import os

# File for persisting tickets
TICKET_FILE = "tickets.csv"

# Initialize ticket storage
if os.path.exists(TICKET_FILE):
    tickets = pd.read_csv(TICKET_FILE, index_col=0)
    ticket_counter = [tickets.index.max() + 1 if not tickets.empty else 0]
else:
    tickets = pd.DataFrame(columns=["Ticket ID", "Description", "Status"])
    ticket_counter = [0]


# Function to save tickets to a file
def save_tickets():
    tickets.to_csv(TICKET_FILE)


# Function to submit a new ticket
def submit_ticket(description):
    ticket_id = ticket_counter[0]
    tickets.loc[ticket_id] = [ticket_id, description, "Open"]
    ticket_counter[0] += 1
    save_tickets()  # Save to file after adding a new ticket
    return f"Ticket #{ticket_id} submitted successfully!"


# Function to view all tickets
def view_tickets():
    if tickets.empty:
        return "No tickets available."
    
    # Create a formatted string with tabs for readability
    output = "Ticket ID\tDescription\t\t\tStatus\n"
    output += "-" * 50 + "\n"
    for _, row in tickets.iterrows():
        output += f"{row['Ticket ID']}\t\t{row['Description']}\t\t{row['Status']}\n"
    return output


# Function to mark a ticket as resolved
def resolve_ticket(ticket_id):
    try:
        ticket_id = int(ticket_id)
        if ticket_id in tickets.index and tickets.at[ticket_id, "Status"] == "Open":
            tickets.at[ticket_id, "Status"] = "Resolved"
            save_tickets()  # Save to file after resolving a ticket
            return f"Ticket #{ticket_id} marked as resolved!"
        return f"Ticket #{ticket_id} not found or already resolved."
    except ValueError:
        return "Please enter a valid numeric Ticket ID."


# Gradio interface components
with gr.Blocks() as ticket_system:
    gr.Markdown("## Simple Problem Ticket System with Data Persistence")
    
    with gr.Tab("Submit Ticket"):
        description_input = gr.Textbox(label="Ticket Description", placeholder="Describe the issue here...")
        submit_button = gr.Button("Submit Ticket")
        submit_output = gr.Textbox(label="Output")
        submit_button.click(submit_ticket, inputs=description_input, outputs=submit_output)
    
    with gr.Tab("View Tickets"):
        view_button = gr.Button("View All Tickets")
        view_output = gr.Textbox(label="Ticket List", lines=15)  # Increase space for better display
        view_button.click(view_tickets, outputs=view_output)
    
    with gr.Tab("Resolve Ticket"):
        resolve_input = gr.Textbox(label="Ticket ID to Resolve", placeholder="Enter Ticket ID...")
        resolve_button = gr.Button("Resolve Ticket")
        resolve_output = gr.Textbox(label="Output")
        resolve_button.click(resolve_ticket, inputs=resolve_input, outputs=resolve_output)

# Launch the Gradio interface
ticket_system.launch(server_name="0.0.0.0", server_port=7865)
