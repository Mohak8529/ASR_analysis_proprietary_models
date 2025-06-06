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
    due_date_str = re.match(r"([A-Za-z]+ \d+)", due_date_str).group(1) if re.match(r"([A-Za-z]+ \d+)", due_date_str) else due_date_str
    call_date = datetime.strptime(call_date_str, "%d/%m/%Y")
    due_date = datetime.strptime(due_date_str, "%B %d")
    due_date = due_date.replace(year=call_date.year)
    if due_date < call_date.replace(month=1, day=1):
        due_date = due_date.replace(year=call_date.year + 1)
    days_overdue = (call_date - due_date).days
    return days_overdue

# Criteria evaluation function
def evaluate_criteria(transcription: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = {
        "id": "2194ae10-33b4-4e8b-9442-e7d77c493ceb",
        "call_date": None,
        "audit_date": "06/05/2025",
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

    # Check for call date in transcription
    call_date_str = None
    for entry in transcription:
        dialogue = entry["dialogue"]
        date_match = re.search(
            r"(?i)(today is|it's|its|the call is on)\s*(\d{1,2}/\d{1,2}/\d{2,4}|[A-Za-z]+ \d{1,2}(?:, \d{4})?)",
            dialogue
        )
        if date_match:
            call_date_str = date_match.group(2)
            if "/" not in call_date_str:
                try:
                    call_date = datetime.strptime(call_date_str, "%B %d, %Y")
                    call_date_str = call_date.strftime("%d/%m/%Y")
                except ValueError:
                    call_date = datetime.strptime(call_date_str + ", 2025", "%B %d, %Y")
                    call_date_str = call_date.strftime("%d/%m/%Y")
            output["call_date"] = call_date_str
            break

    # Extract due date
    due_date_str = None
    for entry in transcription:
        if "due on" in entry["dialogue"].lower():
            due_date_str = entry["dialogue"].split("due on")[1].split(",")[0].strip()
            break

    # Determine type_of_collection
    approach_used = "syntax-based"
    if call_date_str and due_date_str:
        approach_used = "date-based"
        days_overdue = calculate_days_overdue(call_date_str, due_date_str)
        if days_overdue < 0:
            output["type_of_collection"] = "Predues Collection"
        elif days_overdue < 30:
            output["type_of_collection"] = "Postdue Collections Less Than 30 days"
        elif 30 <= days_overdue < 60:
            output["type_of_collection"] = "Postdue Collections Greater Than 30 days"
        else:
            output["type_of_collection"] = "Late Collections Greater Than 60 days"
    else:
        # Use syntactic-cue-based approach
        pre_due_keywords = [
            "is due on", "upcoming payment", "courtesy call to remind", "due soon",
            "next payment is coming", "received your SOA", "help preparing for your due date",
            "ready for the payment", "set for due date", "prepare for your payment"
        ]
        post_due_general_keywords = [
            "was due on", "overdue", "past due", "late payment", "currently overdue",
            "missed the payment deadline", "in arrears", "outstanding balance",
            "failure to pay may result", "payment plan for the overdue"
        ]
        post_due_less_30_keywords = [
            "now overdue", "few days past due", "weeks past due", "late fee if not paid soon",
            "make the payment today or tomorrow", "settle your outstanding balance",
            "as soon as possible"
        ]
        post_due_greater_30_keywords = [
            "over 30 days past due", "more than a month", "seriously delinquent",
            "impact your credit score", "additional fees due to the delay", "escalate this matter",
            "discuss a payment plan", "delinquent status"
        ]
        post_due_greater_60_keywords = [
            "over 60 days past due", "more than two months", "final notice", "sent to collections",
            "reported to credit bureaus", "legal action", "account may be closed",
            "escalated to collections", "full overdue amount", "legal"
        ]

        # High-priority keywords for each category (exact match)
        pre_due_high_priority = ["is due on", "upcoming payment", "due soon"]
        post_due_less_30_high_priority = ["now overdue", "few days past due", "make the payment today"]
        post_due_greater_30_high_priority = ["over 30 days past due", "seriously delinquent", "impact your credit score", "avoid further charges"]
        post_due_greater_60_high_priority = ["legal", "legal action", "reported to credit bureaus"]

        collection_type = None
        for entry in transcription:
            dialogue = entry["dialogue"].lower()
            # Check high-priority keywords with exact matching
            if any(keyword in dialogue for keyword in post_due_greater_60_high_priority):
                collection_type = "Late Collections Greater Than 60 days"
                break
            elif any(keyword in dialogue for keyword in post_due_greater_30_high_priority):
                collection_type = "Postdue Collections Greater Than 30 days"
                break
            elif any(keyword in dialogue for keyword in post_due_less_30_high_priority):
                collection_type = "Postdue Collections Less Than 30 days"
                break
            elif any(keyword in dialogue for keyword in pre_due_high_priority) and not any(keyword in dialogue for keyword in ["overdue", "past due", "late fee", "was due on"]):
                collection_type = "Predues Collection"
                break
            # Check other keywords using fuzzy matching
            elif evaluate_fuzzy_match(dialogue, post_due_greater_60_keywords, threshold=0.5):
                collection_type = "Late Collections Greater Than 60 days"
                break
            elif evaluate_fuzzy_match(dialogue, post_due_greater_30_keywords, threshold=0.5):
                collection_type = "Postdue Collections Greater Than 30 days"
                break
            elif evaluate_fuzzy_match(dialogue, post_due_less_30_keywords, threshold=0.5):
                collection_type = "Postdue Collections Less Than 30 days"
                break
            elif evaluate_fuzzy_match(dialogue, post_due_general_keywords, threshold=0.5):
                collection_type = "Postdue Collections Less Than 30 days"
                break
            elif evaluate_fuzzy_match(dialogue, pre_due_keywords, threshold=0.5) and not any(keyword in dialogue for keyword in ["overdue", "past due", "late fee", "was due on"]):
                collection_type = "Predues Collection"
                break

        # Check "was due on" only if no prior match
        if not collection_type:
            for entry in transcription:
                dialogue = entry["dialogue"].lower()
                if "was due on" in dialogue:
                    if evaluate_fuzzy_match(dialogue, post_due_greater_60_keywords, threshold=0.5):
                        collection_type = "Late Collections Greater Than 60 days"
                    elif evaluate_fuzzy_match(dialogue, post_due_greater_30_keywords, threshold=0.5):
                        collection_type = "Postdue Collections Greater Than 30 days"
                    elif evaluate_fuzzy_match(dialogue, post_due_less_30_keywords, threshold=0.5):
                        collection_type = "Postdue Collections Less Than 30 days"
                    else:
                        collection_type = "Postdue Collections Less Than 30 days"  # Default
                    break

        output["type_of_collection"] = collection_type or "Predues Collection"

    print(f"Used {approach_used} approach for type_of_collection")

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

    # Extract customer details
    for entry in transcription:
        dialogue = entry["dialogue"]
        speaker = entry["speaker"]
        start_time = entry["startTime"]
        end_time = entry["endTime"]

        if speaker == "spk2" and output["customer_name"] is None and ("I'm" in dialogue or "I am" in dialogue):
            name_match = re.search(r"I'm\s+([\w\s]+)|I am\s+([\w\s]+)", dialogue)
            if name_match:
                output["customer_name"] = name_match.group(1) if name_match.group(1) else name_match.group(2)
                output["customer_name"] = output["customer_name"].split("and")[0].strip()
        elif speaker == "spk1" and output["customer_name"] is None and "Miss" in dialogue:
            name_match = re.search(r"Miss\s+([\w\s]+)", dialogue)
            if name_match:
                output["customer_name"] = name_match.group(1).strip()

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

    # Evaluate criteria for Predues Collection
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

            if not criteria_met["did_the_agent_state_the_product_name_current_balance_and_due_date"] and evaluate_fuzzy_match(dialogue, ["is due on", "balance", "loan"]):
                output["did_the_agent_state_the_product_name_current_balance_and_due_date"] = True
                output["did_the_agent_state_the_product_name_current_balance_and_due_date_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Stated product name, balance, and due date", "matched_phrase": "loan balance is due on"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 5)
                criteria_met["did_the_agent_state_the_product_name_current_balance_and_due_date"] = True
            elif evaluate_fuzzy_match(dialogue, ["is due on", "balance", "loan"]):
                output["did_the_agent_state_the_product_name_current_balance_and_due_date_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Stated product name, balance, and due date", "matched_phrase": "loan balance is due on"
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

            if not criteria_met["call_recap"] and evaluate_fuzzy_match(dialogue, ["payment", "is due on"]):
                output["call_recap"] = True
                output["call_recap_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided call recap", "matched_phrase": "payment is due on"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["call_recap"] = True
            elif evaluate_fuzzy_match(dialogue, ["payment", "is due on"]):
                output["call_recap_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided call recap", "matched_phrase": "payment is due on"
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

    # Evaluate criteria for Postdues Collection
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

            if not criteria_met["did_the_agent_state_the_product_name_current_balance_and_due_date"] and evaluate_fuzzy_match(dialogue, ["overdue payment", "credit card", "was due on"]):
                output["did_the_agent_state_the_product_name_current_balance_and_due_date"] = True
                output["did_the_agent_state_the_product_name_current_balance_and_due_date_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Stated product, balance, and due date",
                    "matched_phrase": "overdue payment on your credit card was due on"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 5)
                criteria_met["did_the_agent_state_the_product_name_current_balance_and_due_date"] = True
            elif evaluate_fuzzy_match(dialogue, ["overdue payment", "credit card", "was due on"]):
                output["did_the_agent_state_the_product_name_current_balance_and_due_date_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Stated product, balance, and due date",
                    "matched_phrase": "overdue payment on your credit card was due on"
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

            if not criteria_met["did_the_agent_use_effective_probing_questions"] and evaluate_fuzzy_match(dialogue, ["would you like", "payment"]):
                output["did_the_agent_use_effective_probing_questions"] = True
                output["did_the_agent_use_effective_probing_questions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to make the payment"
                })
                output["negotiation_points"] = str(float(output["negotiation_points"]) + 10)
                criteria_met["did_the_agent_use_effective_probing_questions"] = True
                output["probing_questions_effectiveness"] = True
                output["probing_questions_effectiveness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to make the payment"
                })
            elif evaluate_fuzzy_match(dialogue, ["would you like", "payment"]):
                output["did_the_agent_use_effective_probing_questions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to make the payment"
                })
                output["probing_questions_effectiveness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to make the payment"
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
                    "reason": "Followed policy and procedure", "matched_phrase": "principal and late fees"
                })
                output["process_compliance_points"] = str(float(output["process_compliance_points"]) + 10)
                criteria_met["follow_policies_procedure"] = True
            elif evaluate_fuzzy_match(dialogue, ["due amount", "principal", "late fees"]):
                output["follow_policies_procedure_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Followed policy and procedure", "matched_phrase": "principal and late fees"
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