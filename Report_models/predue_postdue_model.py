#!/usr/bin/env python3.6

import json
from typing import Dict, List, Any
from datetime import datetime
import re

# Load transcription from JSON file
with open('transcription.json', 'r') as file:
    input_data = json.load(file)  # Directly loads as a list

# Fuzzy logic evaluation function
def evaluate_fuzzy_match(dialogue: str, keywords: List[str], threshold: float = 0.5) -> bool:
    dialogue_lower = dialogue.lower()
    keyword_count = sum(1 for keyword in keywords if keyword.lower() in dialogue_lower)
    return keyword_count / len(keywords) >= threshold

# Function to calculate days overdue
def calculate_days_overdue(call_date_str: str, due_date_str: str) -> int:
    due_date_str = due_date_str.replace("th.", "").replace(".", "").strip()
    # Extract only the date part before any additional text
    due_date_str = re.match(r"([A-Za-z]+ \d+)", due_date_str).group(1) if re.match(r"([A-Za-z]+ \d+)", due_date_str) else due_date_str
    due_date_str = re.sub(r'\s+Have\s+you\s+received\s+your\s+SOA\?', '', due_date_str).strip()
    call_date = datetime.strptime(call_date_str, "%d/%m/%Y")
    due_date = datetime.strptime(due_date_str, "%B %d")
    # Set the due date year to the current year, adjust if in the future
    due_date = due_date.replace(year=call_date.year)
    if due_date < call_date.replace(month=1, day=1):  # If due date is in past year, assume next year
        due_date = due_date.replace(year=call_date.year + 1)
    days_overdue = (call_date - due_date).days
    return days_overdue

# Criteria evaluation function
def evaluate_criteria(transcription: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = {
        "id": "2194ae10-33b4-4e8b-9442-e7d77c493ceb",
        "call_date": "06/04/2025",
        "audit_date": "06/04/2025",
        "client": "Maya Bank Collections",
        "customer_name": None,
        "product": "Loan Default",
        "language": "English",
        "agent": None,
        "team_lead": "Alok",
        "qa_lead": "Alok",
        "min_number": None,
        "min_details": [],
        "call_open_timely_manner": False,
        "call_open_timely_manner_details": [],
        "standard_opening_spiel": False,
        "standard_opening_spiel_details": [],
        "did_the_agent_state_the_product_name_current_balance_and_due_date": False,
        "did_the_agent_state_the_product_name_current_balance_and_due_date_details": [],
        "call_opening_points": "0",
        "friendly_confident_tone": False,
        "friendly_confident_tone_details": [],
        "attentive_listening": False,
        "attentive_listening_details": [],
        "customer_experience_points": "0",
        "did_the_agent_use_effective_probing_questions": False,
        "did_the_agent_use_effective_probing_questions_details": [],
        "did_the_agent_act_towards_payment_resolution": False,
        "did_the_agent_act_towards_payment_resolution_details": [],
        "did_the_agent_provide_the_consequence_of_not_paying": False,
        "did_the_agent_provide_the_consequence_of_not_paying_details": [],
        "negotiation_points": "0",
        "follow_policies_procedure": False,
        "follow_policies_procedure_details": [],
        "process_compliance_points": "0",
        "call_document": False,
        "call_document_details": [],
        "documentation_points": "0",
        "call_recap": False,
        "call_recap_details": [],
        "additional_queries": False,
        "additional_queries_details": [],
        "thank_customer": False,
        "thank_customer_details": [],
        "call_closing": False,
        "call_closing_details": [],
        "call_closing_points": "0",
        "call_record_clause": False,
        "call_record_clause_details": [],
        "pid_process": False,
        "pid_process_details": [],
        "udcp_process": False,
        "udcp_process_details": [],
        "call_avoidance": False,
        "call_avoidance_details": [],
        "misleading_information": False,
        "misleading_information_details": [],
        "data_manipulation": False,
        "data_manipulation_details": [],
        "service_compliance_points": "0",
        "probing_questions_effectiveness": False,
        "probing_questions_effectiveness_details": [],
        "payment_resolution_actions": False,
        "payment_resolution_actions_details": [],
        "payment_delay_consequences": False,
        "payment_delay_consequences_details": [],
        "payment_promptness": False,
        "payment_promptness_details": [],
        "customer_verification_accuracy": False,
        "customer_verification_accuracy_details": [],
        "total_points": "0",
        "type_of_collection": None
    }

    due_date_str = None
    for entry in transcription:
        if "due on" in entry["dialogue"]:
            due_date_str = entry["dialogue"].split("due on")[1].split(",")[0].strip()
            break

    if due_date_str:
        days_overdue = calculate_days_overdue(output["call_date"], due_date_str)
        if days_overdue < 0:
            output["type_of_collection"] = "Predues Collection"
        elif days_overdue < 30:
            output["type_of_collection"] = "Postdue Collections Less Than 30 days"
        elif 30 <= days_overdue < 60:
            output["type_of_collection"] = "Postdue Collections Greater Than 30 days"
        else:
            output["type_of_collection"] = "Late Collections Greater Than 60 days"
    else:
        output["type_of_collection"] = "Predues Collection"

    spc_violation = False
    criteria_met = {
        "call_open_timely_manner": False,
        "standard_opening_spiel": False,
        "did_the_agent_state_the_product_name_current_balance_and_due_date": False,
        "friendly_confident_tone": False,
        "attentive_listening": False,
        "did_the_agent_use_effective_probing_questions": False,
        "did_the_agent_act_towards_payment_resolution": False,
        "did_the_agent_provide_the_consequence_of_not_paying": False,
        "follow_policies_procedure": False,
        "call_document": False,
        "call_recap": False,
        "additional_queries": False,
        "thank_customer": False,
        "call_closing": False
    }

    # First pass: Extract customer details with improved logic
    for entry in transcription:
        dialogue = entry["dialogue"]
        speaker = entry["speaker"]
        start_time = entry["startTime"]
        end_time = entry["endTime"]

        # Extract customer name (handle "I'm" or titles)
        if speaker == "spk2" and output["customer_name"] is None and ("I'm" in dialogue or "I am" in dialogue):
            name_match = re.search(r"I'm\s+([\w\s]+)|I am\s+([\w\s]+)", dialogue)
            if name_match:
                output["customer_name"] = name_match.group(1) if name_match.group(1) else name_match.group(2)
                output["customer_name"] = output["customer_name"].split("and")[0].strip()
        elif speaker == "spk1" and output["customer_name"] is None and "Miss" in dialogue:
            name_match = re.search(r"Miss\s+([\w\s]+)", dialogue)
            if name_match:
                output["customer_name"] = name_match.group(1).strip()

        # Extract MIN (handle full or partial numbers)
        if speaker == "spk2" and output["min_number"] is None and "number is" in dialogue:
            min_match = re.search(r"number is\s+(\d+)", dialogue)
            if min_match:
                output["min_number"] = min_match.group(1).strip().replace(".", "")
        elif speaker == "spk1" and output["min_number"] is None and "digits" in dialogue:
            min_match = re.search(r"(\d{4})", dialogue)
            if min_match:
                output["min_number"] = min_match.group(1)

        if speaker == "spk1" and "this is" in dialogue and output["agent"] is None:
            output["agent"] = dialogue.split("this is")[1].split("calling")[0].strip()

    # Second pass: Evaluate criteria for Predues Collection
    if output["type_of_collection"] == "Predues Collection":
        for entry in transcription:
            dialogue = entry["dialogue"]
            speaker = entry["speaker"]
            start_time = entry["startTime"]
            end_time = entry["endTime"]

            if not criteria_met["call_open_timely_manner"] and start_time <= 5 and evaluate_fuzzy_match(dialogue, ["good afternoon", "calling"]):
                output["call_open_timely_manner"] = True
                output["call_open_timely_manner_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call opened within 5 seconds", "matched_phrase": "good afternoon"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 5)
                criteria_met["call_open_timely_manner"] = True
            elif start_time <= 5 and evaluate_fuzzy_match(dialogue, ["good afternoon", "calling"]):
                output["call_open_timely_manner_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call opened within 5 seconds", "matched_phrase": "good afternoon"
                })

            if not criteria_met["standard_opening_spiel"] and evaluate_fuzzy_match(dialogue, ["calling from", "recorded"]):
                output["standard_opening_spiel"] = True
                output["standard_opening_spiel_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used standard opening spiel", "matched_phrase": "calling from Maya Bank"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 10)
                criteria_met["standard_opening_spiel"] = True
            elif evaluate_fuzzy_match(dialogue, ["calling from", "recorded"]):
                output["standard_opening_spiel_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used standard opening spiel", "matched_phrase": "calling from Maya Bank"
                })

            if not criteria_met["did_the_agent_state_the_product_name_current_balance_and_due_date"] and evaluate_fuzzy_match(dialogue, ["due on", "balance", "loan"]):
                output["did_the_agent_state_the_product_name_current_balance_and_due_date"] = True
                output["did_the_agent_state_the_product_name_current_balance_and_due_date_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Stated product name, balance, and due date", "matched_phrase": "loan balance due on April 20th"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 5)
                criteria_met["did_the_agent_state_the_product_name_current_balance_and_due_date"] = True
            elif evaluate_fuzzy_match(dialogue, ["due on", "balance", "loan"]):
                output["did_the_agent_state_the_product_name_current_balance_and_due_date_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Stated product name, balance, and due date", "matched_phrase": "loan balance due on April 20th"
                })

            if not criteria_met["friendly_confident_tone"] and evaluate_fuzzy_match(dialogue, ["thank you", "please"]):
                output["friendly_confident_tone"] = True
                output["friendly_confident_tone_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Maintained friendly and confident tone", "matched_phrase": "thank you"
                })
                output["customer_experience_points"] = str(float(output["customer_experience_points"]) + 10)
                criteria_met["friendly_confident_tone"] = True
            elif evaluate_fuzzy_match(dialogue, ["thank you", "please"]):
                output["friendly_confident_tone_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Maintained friendly and confident tone", "matched_phrase": "thank you"
                })

            if not criteria_met["attentive_listening"] and evaluate_fuzzy_match(dialogue, ["of course", "clarification"]):
                output["attentive_listening"] = True
                output["attentive_listening_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Listened attentively and displayed understanding", "matched_phrase": "of course"
                })
                output["customer_experience_points"] = str(float(output["customer_experience_points"]) + 10)
                criteria_met["attentive_listening"] = True
            elif evaluate_fuzzy_match(dialogue, ["of course", "clarification"]):
                output["attentive_listening_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Listened attentively and displayed understanding", "matched_phrase": "of course"
                })

            if not criteria_met["did_the_agent_use_effective_probing_questions"] and evaluate_fuzzy_match(dialogue, ["would you like", "receive"]):
                output["did_the_agent_use_effective_probing_questions"] = True
                output["did_the_agent_use_effective_probing_questions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to receive your SOA"
                })
                output["negotiation_points"] = str(float(output["negotiation_points"]) + 10)
                criteria_met["did_the_agent_use_effective_probing_questions"] = True
                output["probing_questions_effectiveness"] = True
                output["probing_questions_effectiveness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to receive your SOA"
                })
            elif evaluate_fuzzy_match(dialogue, ["would you like", "receive"]):
                output["did_the_agent_use_effective_probing_questions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to receive your SOA"
                })
                output["probing_questions_effectiveness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to receive your SOA"
                })

            if not criteria_met["follow_policies_procedure"] and evaluate_fuzzy_match(dialogue, ["benefits", "consequences"]):
                output["follow_policies_procedure"] = True
                output["follow_policies_procedure_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Followed policies and procedures", "matched_phrase": "benefits of paying on time"
                })
                output["process_compliance_points"] = str(float(output["process_compliance_points"]) + 15)
                criteria_met["follow_policies_procedure"] = True
            elif evaluate_fuzzy_match(dialogue, ["benefits", "consequences"]):
                output["follow_policies_procedure_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Followed policies and procedures", "matched_phrase": "benefits of paying on time"
                })

            if not criteria_met["call_document"] and evaluate_fuzzy_match(dialogue, ["document", "tools"]):
                output["call_document"] = True
                output["call_document_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Properly documented the call", "matched_phrase": "documented in tools"
                })
                output["documentation_points"] = str(float(output["documentation_points"]) + 10)
                criteria_met["call_document"] = True
            elif evaluate_fuzzy_match(dialogue, ["document", "tools"]):
                output["call_document_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Properly documented the call", "matched_phrase": "documented in tools"
                })

            if not criteria_met["call_recap"] and evaluate_fuzzy_match(dialogue, ["payment", "due date"]):
                output["call_recap"] = True
                output["call_recap_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided call recap", "matched_phrase": "payment due on April 20th"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["call_recap"] = True
            elif evaluate_fuzzy_match(dialogue, ["payment", "due date"]):
                output["call_recap_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided call recap", "matched_phrase": "payment due on April 20th"
                })

            if not criteria_met["additional_queries"] and evaluate_fuzzy_match(dialogue, ["anything else", "questions"]):
                output["additional_queries"] = True
                output["additional_queries_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Asked if there are additional questions", "matched_phrase": "Anything else I can assist with"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["additional_queries"] = True
            elif evaluate_fuzzy_match(dialogue, ["anything else", "questions"]):
                output["additional_queries_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Asked if there are additional questions", "matched_phrase": "Anything else I can assist with"
                })

            if not criteria_met["thank_customer"] and evaluate_fuzzy_match(dialogue, ["thank you", "great day"]):
                output["thank_customer"] = True
                output["thank_customer_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Thanked the customer", "matched_phrase": "Thank you and have a great day"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["thank_customer"] = True
            elif evaluate_fuzzy_match(dialogue, ["thank you", "great day"]):
                output["thank_customer_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Thanked the customer", "matched_phrase": "Thank you and have a great day"
                })

            if not criteria_met["call_closing"] and evaluate_fuzzy_match(dialogue, ["goodbye", "day ahead"]):
                output["call_closing"] = True
                output["call_closing_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call closed professionally", "matched_phrase": "goodbye"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["call_closing"] = True
            elif evaluate_fuzzy_match(dialogue, ["goodbye", "day ahead"]):
                output["call_closing_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call closed professionally", "matched_phrase": "goodbye"
                })

            if evaluate_fuzzy_match(dialogue, ["recorded", "quality"]):
                output["call_record_clause"] = True
                output["call_record_clause_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call record clause mentioned", "matched_phrase": "This call is recorded"
                })

            if evaluate_fuzzy_match(dialogue, ["full name", "mobile number"]):
                output["pid_process"] = True
                output["pid_process_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "PID process followed", "matched_phrase": "full name and registered mobile number"
                })

            if not evaluate_fuzzy_match(dialogue, ["swear", "offensive"]):
                output["udcp_process"] = True
                output["udcp_process_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Adhered to UDCP Prohibition", "matched_phrase": "No offensive language"
                })

            if evaluate_fuzzy_match(dialogue, ["avoid", "hang up"]):
                output["call_avoidance"] = True
                output["call_avoidance_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call avoidance detected", "matched_phrase": "avoid"
                })
                spc_violation = True

            if evaluate_fuzzy_match(dialogue, ["mislead", "false"]):
                output["misleading_information"] = True
                output["misleading_information_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Misleading information detected", "matched_phrase": "mislead"
                })
                spc_violation = True

            if evaluate_fuzzy_match(dialogue, ["manipulate", "alter"]):
                output["data_manipulation"] = True
                output["data_manipulation_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Data manipulation detected", "matched_phrase": "manipulate"
                })
                spc_violation = True

            if output["customer_name"] and output["min_number"] and speaker == "spk2" and output["customer_name"] in dialogue and output["min_number"] in dialogue:
                output["customer_verification_accuracy"] = True
                output["customer_verification_accuracy_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Accurate verification provided", "matched_phrase": f"{output['customer_name']} {output['min_number']}"
                })

    # Second pass: Evaluate criteria for Postdues Collection
    elif output["type_of_collection"] in ["Postdue Collections Less Than 30 days", "Postdue Collections Greater Than 30 days", "Late Collections Greater Than 60 days"]:
        for entry in transcription:
            dialogue = entry["dialogue"]
            speaker = entry["speaker"]
            start_time = entry["startTime"]
            end_time = entry["endTime"]

            if not criteria_met["call_open_timely_manner"] and start_time <= 5 and evaluate_fuzzy_match(dialogue, ["good afternoon", "calling"]):
                output["call_open_timely_manner"] = True
                output["call_open_timely_manner_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call opened within 5 seconds", "matched_phrase": "good afternoon"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 3)
                criteria_met["call_open_timely_manner"] = True
            elif start_time <= 5 and evaluate_fuzzy_match(dialogue, ["good afternoon", "calling"]):
                output["call_open_timely_manner_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call opened within 5 seconds", "matched_phrase": "good afternoon"
                })

            if not criteria_met["standard_opening_spiel"] and evaluate_fuzzy_match(dialogue, ["calling from", "recorded"]):
                output["standard_opening_spiel"] = True
                output["standard_opening_spiel_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used standard opening spiel", "matched_phrase": "calling from FinTrust Services"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 5)
                criteria_met["standard_opening_spiel"] = True
            elif evaluate_fuzzy_match(dialogue, ["calling from", "recorded"]):
                output["standard_opening_spiel_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used standard opening spiel", "matched_phrase": "calling from FinTrust Services"
                })

            if not criteria_met["did_the_agent_state_the_product_name_current_balance_and_due_date"] and evaluate_fuzzy_match(dialogue, ["overdue payment", "credit card", "due on"]):
                output["did_the_agent_state_the_product_name_current_balance_and_due_date"] = True
                output["did_the_agent_state_the_product_name_current_balance_and_due_date_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Stated product, balance, and due date",
                    "matched_phrase": "overdue payment of ₹3,200 on your credit card ending with 7890, which was due on April 5th"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 5)
                criteria_met["did_the_agent_state_the_product_name_current_balance_and_due_date"] = True
            elif evaluate_fuzzy_match(dialogue, ["overdue payment", "credit card", "due on"]):
                output["did_the_agent_state_the_product_name_current_balance_and_due_date_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Stated product, balance, and due date",
                    "matched_phrase": "overdue payment of ₹3,200 on your credit card ending with 7890, which was due on April 5th"
                })

            if not criteria_met["friendly_confident_tone"] and evaluate_fuzzy_match(dialogue, ["thank you", "please"]):
                output["friendly_confident_tone"] = True
                output["friendly_confident_tone_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Maintained friendly and confident tone", "matched_phrase": "thank you"
                })
                output["customer_experience_points"] = str(float(output["customer_experience_points"]) + 7)
                criteria_met["friendly_confident_tone"] = True
            elif evaluate_fuzzy_match(dialogue, ["thank you", "please"]):
                output["friendly_confident_tone_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Maintained friendly and confident tone", "matched_phrase": "thank you"
                })

            if not criteria_met["attentive_listening"] and evaluate_fuzzy_match(dialogue, ["of course", "clarification"]):
                output["attentive_listening"] = True
                output["attentive_listening_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Listened attentively and displayed understanding", "matched_phrase": "of course"
                })
                output["customer_experience_points"] = str(float(output["customer_experience_points"]) + 7)
                criteria_met["attentive_listening"] = True
            elif evaluate_fuzzy_match(dialogue, ["of course", "clarification"]):
                output["attentive_listening_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Listened attentively and displayed understanding", "matched_phrase": "of course"
                })

            if not criteria_met["did_the_agent_use_effective_probing_questions"] and evaluate_fuzzy_match(dialogue, ["would you like", "anything else"]):
                output["did_the_agent_use_effective_probing_questions"] = True
                output["did_the_agent_use_effective_probing_questions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to make the payment today"
                })
                output["negotiation_points"] = str(float(output["negotiation_points"]) + 10)
                criteria_met["did_the_agent_use_effective_probing_questions"] = True
                output["probing_questions_effectiveness"] = True
                output["probing_questions_effectiveness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to make the payment today"
                })
            elif evaluate_fuzzy_match(dialogue, ["would you like", "anything else"]):
                output["did_the_agent_use_effective_probing_questions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to make the payment today"
                })
                output["probing_questions_effectiveness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to make the payment today"
                })

            if not criteria_met["did_the_agent_act_towards_payment_resolution"] and evaluate_fuzzy_match(dialogue, ["payment", "today", "evening"]):
                output["did_the_agent_act_towards_payment_resolution"] = True
                output["did_the_agent_act_towards_payment_resolution_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Acted towards payment resolution", "matched_phrase": "pay it online by this evening"
                })
                output["negotiation_points"] = str(float(output["negotiation_points"]) + 20)
                criteria_met["did_the_agent_act_towards_payment_resolution"] = True
                output["payment_resolution_actions"] = True
                output["payment_resolution_actions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Payment resolution agreed", "matched_phrase": "pay it online by this evening"
                })
                output["payment_promptness"] = True
                output["payment_promptness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Customer committed to timely payment", "matched_phrase": "pay it online by this evening"
                })
            elif evaluate_fuzzy_match(dialogue, ["payment", "today", "evening"]):
                output["did_the_agent_act_towards_payment_resolution_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Acted towards payment resolution", "matched_phrase": "pay it online by this evening"
                })
                output["payment_resolution_actions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Payment resolution agreed", "matched_phrase": "pay it online by this evening"
                })
                output["payment_promptness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Customer committed to timely payment", "matched_phrase": "pay it online by this evening"
                })

            if not criteria_met["did_the_agent_provide_the_consequence_of_not_paying"] and evaluate_fuzzy_match(dialogue, ["late fee", "increase"]):
                output["did_the_agent_provide_the_consequence_of_not_paying"] = True
                output["did_the_agent_provide_the_consequence_of_not_paying_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided consequence of not paying", "matched_phrase": "late fee will not increase further"
                })
                output["negotiation_points"] = str(float(output["negotiation_points"]) + 10)
                criteria_met["did_the_agent_provide_the_consequence_of_not_paying"] = True
                output["payment_delay_consequences"] = True
                output["payment_delay_consequences_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Explained delay consequence", "matched_phrase": "late fee will not increase further"
                })
            elif evaluate_fuzzy_match(dialogue, ["late fee", "increase"]):
                output["did_the_agent_provide_the_consequence_of_not_paying_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided consequence of not paying", "matched_phrase": "late fee will not increase further"
                })
                output["payment_delay_consequences_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Explained delay consequence", "matched_phrase": "late fee will not increase further"
                })

            if not criteria_met["follow_policies_procedure"] and evaluate_fuzzy_match(dialogue, ["due amount", "principal", "late fees"]):
                output["follow_policies_procedure"] = True
                output["follow_policies_procedure_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Followed policy and procedure", "matched_phrase": "₹3,000 as the principal and ₹200 as late fees"
                })
                output["process_compliance_points"] = str(float(output["process_compliance_points"]) + 10)
                criteria_met["follow_policies_procedure"] = True
            elif evaluate_fuzzy_match(dialogue, ["due amount", "principal", "late fees"]):
                output["follow_policies_procedure_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Followed policy and procedure", "matched_phrase": "₹3,000 as the principal and ₹200 as late fees"
                })

            if not criteria_met["call_document"] and evaluate_fuzzy_match(dialogue, ["document", "tools"]):
                output["call_document"] = True
                output["call_document_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Properly documented the call", "matched_phrase": "documented in tools"
                })
                output["documentation_points"] = str(float(output["documentation_points"]) + 5)
                criteria_met["call_document"] = True
            elif evaluate_fuzzy_match(dialogue, ["document", "tools"]):
                output["call_document_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Properly documented the call", "matched_phrase": "documented in tools"
                })

            if not criteria_met["call_recap"] and evaluate_fuzzy_match(dialogue, ["late fee", "increase"]):
                output["call_recap"] = True
                output["call_recap_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided call recap", "matched_phrase": "late fee will not increase further"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["call_recap"] = True
            elif evaluate_fuzzy_match(dialogue, ["late fee", "increase"]):
                output["call_recap_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided call recap", "matched_phrase": "late fee will not increase further"
                })

            if not criteria_met["additional_queries"] and evaluate_fuzzy_match(dialogue, ["anything else", "help"]):
                output["additional_queries"] = True
                output["additional_queries_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Asked if there are additional queries", "matched_phrase": "Is there anything else I can help you with"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 3)
                criteria_met["additional_queries"] = True
            elif evaluate_fuzzy_match(dialogue, ["anything else", "help"]):
                output["additional_queries_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Asked if there are additional queries", "matched_phrase": "Is there anything else I can help you with"
                })

            if not criteria_met["thank_customer"] and evaluate_fuzzy_match(dialogue, ["thank you", "great day"]):
                output["thank_customer"] = True
                output["thank_customer_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Thanked the customer", "matched_phrase": "Thank you for your time"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 2)
                criteria_met["thank_customer"] = True
            elif evaluate_fuzzy_match(dialogue, ["thank you", "great day"]):
                output["thank_customer_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Thanked the customer", "matched_phrase": "Thank you for your time"
                })

            if not criteria_met["call_closing"] and evaluate_fuzzy_match(dialogue, ["goodbye", "day ahead"]):
                output["call_closing"] = True
                output["call_closing_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call closed professionally", "matched_phrase": "goodbye"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["call_closing"] = True
            elif evaluate_fuzzy_match(dialogue, ["goodbye", "day ahead"]):
                output["call_closing_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call closed professionally", "matched_phrase": "goodbye"
                })

            if evaluate_fuzzy_match(dialogue, ["recorded", "quality"]):
                output["call_record_clause"] = True
                output["call_record_clause_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call record clause mentioned", "matched_phrase": "This call is recorded"
                })

            if evaluate_fuzzy_match(dialogue, ["full name", "mobile number"]):
                output["pid_process"] = True
                output["pid_process_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "PID process followed", "matched_phrase": "full name and registered mobile number"
                })

            if not evaluate_fuzzy_match(dialogue, ["swear", "offensive"]):
                output["udcp_process"] = True
                output["udcp_process_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Adhered to UDCP Prohibition", "matched_phrase": "No offensive language"
                })

            if evaluate_fuzzy_match(dialogue, ["avoid", "hang up"]):
                output["call_avoidance"] = True
                output["call_avoidance_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call avoidance detected", "matched_phrase": "avoid"
                })
                spc_violation = True

            if evaluate_fuzzy_match(dialogue, ["mislead", "false"]):
                output["misleading_information"] = True
                output["misleading_information_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Misleading information detected", "matched_phrase": "mislead"
                })
                spc_violation = True

            if evaluate_fuzzy_match(dialogue, ["manipulate", "alter"]):
                output["data_manipulation"] = True
                output["data_manipulation_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Data manipulation detected", "matched_phrase": "manipulate"
                })
                spc_violation = True

            if output["customer_name"] and output["min_number"] and speaker == "spk2" and output["customer_name"] in dialogue and output["min_number"] in dialogue:
                output["customer_verification_accuracy"] = True
                output["customer_verification_accuracy_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Accurate verification provided", "matched_phrase": f"{output['customer_name']} {output['min_number']}"
                })

    if spc_violation or not (output["call_record_clause"] and output["pid_process"] and output["udcp_process"]):
        output["total_points"] = "0"
    else:
        total_points = (float(output["call_opening_points"]) +
                        float(output["customer_experience_points"]) +
                        float(output["negotiation_points"]) +
                        float(output["process_compliance_points"]) +
                        float(output["documentation_points"]) +
                        float(output["call_closing_points"]) +
                        float(output["service_compliance_points"]))
        output["total_points"] = str(total_points)

    return output

if __name__ == "__main__":
    transcription = input_data  
    result = evaluate_criteria(transcription)
    print(json.dumps(result, indent=4))

    with open('output.json', 'w') as outfile:
        json.dump(result, outfile, indent=4)