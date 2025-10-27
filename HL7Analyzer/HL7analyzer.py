import re

def create_message_dictionary(HL7_message_file):
    """This function will take the message file specified by the user and create a dictionary."""
    
    # Open the HL7 message file and assign it to an object.
    with open(HL7_message_file) as file_object:
        HL7_message_import = file_object.read()

    # Define the field delimiter and initialize the HL7 message dictionary.
    HL7_message_delimiter = HL7_message_import[3]
    HL7_message_dictionary = {}
    
    # Parse the import text into a list of segments.
    HL7_message_segments_list = HL7_message_import.split("\n")

    # Split each segment into its fields and add them to the dictionary
    for segment in HL7_message_segments_list:
        segment_name = segment[0:3]
        segment_pos = 1
        segment_parse = segment.split(HL7_message_delimiter)
        for field in segment_parse:
            HL7_message_dictionary[f"{segment_name}-{segment_pos}"] = field
            segment_pos += 1

    # Return the HL7 message dictionary
    return HL7_message_dictionary


def view_message_field(HL7_message_dictionary):
    """This function takes user input of the form 'Segment-Field#' (e.g. MSH-9) and outputs the contents of that field."""
    
    # Ask the user which field they wish to evaluate the contents of then return its contents. Enter 'q' to stop.
    field_check_prompt = "\nEnter the name of a field you wish to see the contents of:\nEnter 'q' to stop.\n"
    field_check_prompt += "--> "
    field_check = ""
    while field_check != "q":
        field_check = input(field_check_prompt)

        if field_check == "q":
            print("Returning to the main menu...\n")
        
        else:
            try:
                print(HL7_message_dictionary[field_check])
            except KeyError:
                print("This field does not exist.\n")

def analyze_message(HL7_message_dictionary):
    """This function will check the message type and then perform a series of steps to summarize the message."""
    
    # Get the message type from MSH-9 by extracting the first 7 nonwhitespace characters.    
    message_type_re = re.search(r"(\S{7})",HL7_message_dictionary['MSH-9'])
    message_type = message_type_re.group(1)

    if message_type == "ADT^A01":
        # Get key data from message to generate a summary of the event.
        location = HL7_message_dictionary['MSH-4']
        # patient_name_re = re.search(r"(.+?)\^+(.+?)\^+(.+?)\^+(.+?)$",HL7_message_dictionary['PID-6'])
        patient_name_re = re.search(r"([a-zA-Z]{0,})\^+([a-zA-Z]{0,})\^+([a-zA-Z]{0,})\^+([a-zA-Z]{0,})$",HL7_message_dictionary['PID-6'])
        patient_name = f"{patient_name_re.group(2)} {patient_name_re.group(3)} {patient_name_re.group(1)} {patient_name_re.group(4)}"
        patient_id_re = re.search(r"(.+?)\^",HL7_message_dictionary['PID-4'])
        patient_id = patient_id_re.group(1)
        admission_datetime_re = re.search(r"(\d{4})+(\d{2})+(\d{2})+(\d{4})",HL7_message_dictionary['EVN-3'])
        admission_datetime = f"{admission_datetime_re.group(2)}/{admission_datetime_re.group(3)}/{admission_datetime_re.group(1)} {admission_datetime_re.group(4)}"

        # Format output message.
        output = f"\nADT A01 - Patient admission event\n"
        output += f"Admission Date/Time: {admission_datetime}\n"
        output += f"Admission Location: {location}\n"
        output += f"Patient Name: {patient_name}\n"
        output += f"Patient ID: {patient_id}\n"

    elif message_type == "ADT^A03":
        # Get key data from message to generate a summary of the event.
        # patient_name_re = re.search(r"(.+?)\^+(.+?)\^+(.+?)\^+(.+?)$",HL7_message_dictionary['PID-6'])
        patient_name_re = re.search(r"([a-zA-Z]{0,})\^+([a-zA-Z]{0,})\^+([a-zA-Z]{0,})\^+([a-zA-Z]{0,})$",HL7_message_dictionary['PID-6'])
        patient_name = f"{patient_name_re.group(2)} {patient_name_re.group(3)} {patient_name_re.group(1)} {patient_name_re.group(4)}"
        patient_id_re = re.search(r"(.+?)\^",HL7_message_dictionary['PID-4'])
        patient_id = patient_id_re.group(1)
        discharge_datetime_re = re.search(r"(\d{4})+(\d{2})+(\d{2})+(\d{4})",HL7_message_dictionary['EVN-3'])
        discharge_datetime = f"{discharge_datetime_re.group(2)}/{discharge_datetime_re.group(3)}/{discharge_datetime_re.group(1)} {discharge_datetime_re.group(4)}"

        # Format output message.
        output = f"\nADT A03 - Patient discharge event\n"
        output += f"Discharge Date/Time: {discharge_datetime}\n"
        output += f"Patient Name: {patient_name}\n"
        output += f"Patient ID: {patient_id}"

    elif message_type == "RDE^O11":
        # Get key data from message to generate a summary of the event.
        ordered_datetime_re = re.search(r"(\d{4})+(\d{2})+(\d{2})+(\d{4})+(\d{2})",HL7_message_dictionary['ORC-10'])
        ordered_datetime = f"{ordered_datetime_re.group(2)}/{ordered_datetime_re.group(3)}/{ordered_datetime_re.group(1)} {ordered_datetime_re.group(4)}"
        txstart_datetime_re = re.search(r"(\d{4})+(\d{2})+(\d{2})+(\d{4})+(\d{2})",HL7_message_dictionary['TQ1-8'])
        txstart_datetime = f"{txstart_datetime_re.group(2)}/{txstart_datetime_re.group(3)}/{txstart_datetime_re.group(1)} {txstart_datetime_re.group(4)}"
        txend_datetime_re = re.search(r"(\d{4})+(\d{2})+(\d{2})+(\d{4})+(\d{2})",HL7_message_dictionary['TQ1-9'])
        txend_datetime = f"{txend_datetime_re.group(2)}/{txend_datetime_re.group(3)}/{txend_datetime_re.group(1)} {txend_datetime_re.group(4)}"
        # patient_name_re = re.search(r"(.+?)\^+(.+?)\^+(.+?)\^+([a-zA-Z]{0,})$",HL7_message_dictionary['PID-6'])
        patient_name_re = re.search(r"([a-zA-Z]{0,})\^+([a-zA-Z]{0,})\^+([a-zA-Z]{0,})\^+([a-zA-Z]{0,})$",HL7_message_dictionary['PID-6'])
        patient_name = f"{patient_name_re.group(2)} {patient_name_re.group(3)} {patient_name_re.group(1)} {patient_name_re.group(4)}"
        patient_id_re = re.search(r"(.+?)\^",HL7_message_dictionary['PID-4'])
        patient_id = patient_id_re.group(1)
        medication_re = re.search(r"\^{2,}(.+?)$",HL7_message_dictionary['RXE-3'])
        medication = medication_re.group(1)
        med_route_re = re.search(r"\^(.+?)$", HL7_message_dictionary['RXR-2'])
        med_route = med_route_re.group(1)
        med_frequency = HL7_message_dictionary['TQ1-4']
        med_class = HL7_message_dictionary['ZTA-6']

        # Format output message.
        output = f"\nRDE O11 - Pharmacy/Treatment encoded order\n"
        output += f"Ordered Date/Time: {ordered_datetime}\n"
        output += f"Treatment Start Date/Time: {txstart_datetime}\n"
        output += f"Treatment End Date/Time: {txend_datetime}\n"
        output += f"Patient Name: {patient_name}\n"
        output += f"Patient ID: {patient_id}\n"
        output += f"Medication: {medication} {med_route} {med_frequency}\n"
        output += f"Medication Class: {med_class}\n"

    else:
        output ="\nThis message type is not recognized."
    
    print(output)


def main_loop():
    """This is the main program loop that users will use to evaluate HL7 messages."""

    # Initialize variables and define prompts.
    action = ""
    
    action_prompt = "\n### Main Menu ###"
    action_prompt += "\nWhat function would you like to perform?\n"
    action_prompt += "Enter 'v' to view the contents of a field within a message.\n"
    action_prompt += "Enter 'a' to analyze an entire message.\n"
    action_prompt += "Enter 'q' to end the program.\n"
    action_prompt += "--> "
    
    message_prompt = "\nEnter the name of the message file to be processed:\n"
    message_prompt += "--> "
    

    # Ask the user for the file to be opened.
    while action != "q":
        
        # Ask the user for the action they wish to perform.
        action = input(action_prompt)

        if action == "v":
            HL7_message_file = input(message_prompt)
            try:
                HL7_message_dictionary = create_message_dictionary(HL7_message_file)
                view_message_field(HL7_message_dictionary)
            except:
                print(f"An error has been encountered.\n")
        
        elif action == "a":
            HL7_message_file = input(message_prompt)
            try:
                HL7_message_dictionary = create_message_dictionary(HL7_message_file)
                analyze_message(HL7_message_dictionary)
            except:
                print("An error has been encountered.\n")
        
        elif action == "q":
            print("Exiting program...")
            break

        else:
            print("The command you entered is invalid. Please try again.\n")

main_loop()


