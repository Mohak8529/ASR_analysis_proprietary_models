#!/usr/bin/env python3.6

import json
from typing import Dict, List, Any
from datetime import datetime
import re
from fuzzywuzzy import fuzz

# Load transcription from JSON file
with open('transcription.json', 'r') as file:
    input_data = json.load(file)

# Load criteria from JSON files
with open('predues_criteria.json', 'r') as file:
    predues_criteria = json.load(file)["criteria"]

with open('postdues_criteria.json', 'r') as file:
    postdues_criteria = json.load(file)["criteria"]

# Enhanced fuzzy match function using fuzzywuzzy
def evaluate_fuzzy_match(dialogue: str, keywords: List[str], threshold: float = 0.85) -> bool:
    dialogue_lower = dialogue.lower()
    for keyword in keywords:
        if fuzz.partial_ratio(keyword.lower(), dialogue_lower) >= (threshold * 100):
            return True
    return False

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

    # Extract call date
    call_date_str = None
    for entry in transcription:
        dialogue = entry["dialogue"]
        date_match = re.search(
            r"(?i)(today is|it's|its|the call is on)\s*(\d{1,2}/\d{1,2}/\d{2,4}|[A-Za-z]+ \d{1,2}(?:, \d{4})?)",
            dialogue
        )
        valid_months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        if date_match:
            call_date_str = date_match.group(2)
            if "/" not in call_date_str:
                if any(month in call_date_str.lower() for month in valid_months):
                    try:
                        call_date = datetime.strptime(call_date_str, "%B %d, %Y")
                        call_date_str = call_date.strftime("%d/%m/%Y")
                    except ValueError:
                        try:
                            call_date = datetime.strptime(call_date_str + ", 2025", "%B %d, %Y")
                            call_date_str = call_date.strftime("%d/%m/%Y")
                        except ValueError:
                            continue
                else:
                    continue
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
        pre_due_keywords = [
            "is due on", "upcoming payment", "courtesy call to remind", "due soon",
            "next payment is coming", "received your SOA", "help preparing for your due date",
            "ready for the payment", "set for due date", "prepare for your payment", "due next week"
        ]
        post_due_less_30_keywords = [
            "now overdue", "few days past due", "weeks past due", "late fee if not paid soon",
            "make the payment today or tomorrow", "settle your outstanding balance",
            "as soon as possible", "one day past due"
        ]
        post_due_greater_30_keywords = [
            "over 30 days past due", "more than a month", "seriously delinquent",
            "impact your credit score", "additional fees due to the delay", "escalate this matter",
            "discuss a payment plan", "delinquent status", "month overdue"
        ]
        post_due_greater_60_keywords = [
            "over 60 days past due", "more than two months", "final notice",
            "reported to credit bureaus", "legal action", "account may be closed",
            "full overdue amount", "legal", "two months late"
        ]

        pre_due_high_priority = ["is due on", "upcoming payment", "due soon"]
        post_due_less_30_high_priority = ["now overdue", "few days past due", "make the payment today"]
        post_due_greater_30_high_priority = ["over 30 days past due", "seriously delinquent", "impact your credit score"]
        post_due_greater_60_high_priority = ["legal", "legal action", "reported to credit bureaus", "blacklist"]

        collection_type = None
        for entry in transcription:
            dialogue = entry["dialogue"].lower()
            if any(fuzz.partial_ratio(keyword, dialogue) >= 85 for keyword in post_due_greater_60_high_priority):
                collection_type = "Late Collections Greater Than 60 days"
                break
            elif any(fuzz.partial_ratio(keyword, dialogue) >= 85 for keyword in post_due_greater_30_high_priority):
                collection_type = "Postdue Collections Greater Than 30 days"
                break
            elif any(fuzz.partial_ratio(keyword, dialogue) >= 85 for keyword in post_due_less_30_high_priority):
                collection_type = "Postdue Collections Less Than 30 days"
                break
            elif any(fuzz.partial_ratio(keyword, dialogue) >= 85 for keyword in pre_due_high_priority) and not any(keyword in dialogue for keyword in ["overdue", "past due", "late fee", "was due on"]):
                collection_type = "Predues Collection"
                break
            elif evaluate_fuzzy_match(dialogue, post_due_greater_60_keywords):
                collection_type = "Late Collections Greater Than 60 days"
                break
            elif evaluate_fuzzy_match(dialogue, post_due_greater_30_keywords):
                collection_type = "Postdue Collections Greater Than 30 days"
                break
            elif evaluate_fuzzy_match(dialogue, post_due_less_30_keywords):
                collection_type = "Postdue Collections Less Than 30 days"
                break
            elif evaluate_fuzzy_match(dialogue, pre_due_keywords) and not any(keyword in dialogue for keyword in ["overdue", "past due", "late fee", "was due on"]):
                collection_type = "Predues Collection"
                break

        if not collection_type:
            for entry in transcription:
                dialogue = entry["dialogue"].lower()
                if "was due on" in dialogue:
                    if evaluate_fuzzy_match(dialogue, post_due_greater_60_keywords):
                        collection_type = "Late Collections Greater Than 60 days"
                    elif evaluate_fuzzy_match(dialogue, post_due_greater_30_keywords):
                        collection_type = "Postdue Collections Greater Than 30 days"
                    elif evaluate_fuzzy_match(dialogue, post_due_less_30_keywords):
                        collection_type = "Postdue Collections Less Than 30 days"
                    else:
                        collection_type = "Postdue Collections Less Than 30 days"
                    break

        output["type_of_collection"] = collection_type or "Unknown"

    print(f"Used {approach_used} approach for type_of_collection")

    # Select criteria based on collection type
    criteria = predues_criteria if output["type_of_collection"] == "Predues Collection" else postdues_criteria

    spc_violation = False
    criteria_met = {criterion["name"]: False for criterion in criteria}

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

    # Evaluate criteria dynamically
    for entry in transcription:
        dialogue = entry["dialogue"]
        speaker = entry["speaker"]
        start_time = entry["startTime"]
        end_time = entry["endTime"]

        for criterion in criteria:
            criterion_name = criterion["name"]
            keywords = criterion["keywords"]
            max_score = criterion["max_score"]
            category = criterion["category"]
            details_reason = criterion["details_reason"]
            matched_phrase = criterion["matched_phrase"]
            condition = criterion.get("condition", "")
            negation = criterion.get("negation", False)
            violation = criterion.get("violation", False)
            custom_condition = criterion.get("custom_condition", "")
            additional_fields = criterion.get("additional_fields", [])

            # Check if criterion is met
            is_met = False
            if custom_condition:
                # Handle custom condition for customer_verification_accuracy
                if criterion_name == "customer_verification_accuracy":
                    is_met = output["customer_name"] and output["min_number"] and speaker == "spk2" and output["customer_name"] in dialogue and output["min_number"] in dialogue
            else:
                # Check condition for criteria like call_open_timely_manner
                condition_met = True
                if condition == "start_time <= 5":
                    condition_met = start_time <= 5
                # Check keywords with or without negation
                keyword_match = evaluate_fuzzy_match(dialogue, keywords)
                is_met = condition_met and (not keyword_match if negation else keyword_match)

            # Update output if criterion is met and not already marked
            if is_met and not criteria_met[criterion_name]:
                output[criterion_name] = True
                output[f"{criterion_name}_details"].append({
                    "dialogue": dialogue,
                    "speaker": speaker,
                    "startTime": start_time,
                    "endTime": end_time,
                    "reason": details_reason,
                    "matched_phrase": matched_phrase.replace("customer_name min_number", f"{output['customer_name']} {output['min_number']}" if criterion_name == "customer_verification_accuracy" else matched_phrase)
                })
                if max_score > 0:
                    output[category] = str(float(output[category]) + max_score)
                criteria_met[criterion_name] = True
                if violation:
                    spc_violation = True
                # Handle additional fields
                for additional in additional_fields:
                    additional_name = additional["name"]
                    output[additional_name] = True
                    output[f"{additional_name}_details"].append({
                        "dialogue": dialogue,
                        "speaker": speaker,
                        "startTime": start_time,
                        "endTime": end_time,
                        "reason": additional["details_reason"],
                        "matched_phrase": additional["matched_phrase"]
                    })
            elif is_met or (not negation and evaluate_fuzzy_match(dialogue, keywords)):
                # Append details even if criterion was already met
                output[f"{criterion_name}_details"].append({
                    "dialogue": dialogue,
                    "speaker": speaker,
                    "startTime": start_time,
                    "endTime": end_time,
                    "reason": details_reason,
                    "matched_phrase": matched_phrase.replace("customer_name min_number", f"{output['customer_name']} {output['min_number']}" if criterion_name == "customer_verification_accuracy" else matched_phrase)
                })
                if violation:
                    spc_violation = True
                # Append details for additional fields
                for additional in additional_fields:
                    output[f"{additional['name']}_details"].append({
                        "dialogue": dialogue,
                        "speaker": speaker,
                        "startTime": start_time,
                        "endTime": end_time,
                        "reason": additional["details_reason"],
                        "matched_phrase": additional["matched_phrase"]
                    })

    # Calculate total points
    if spc_violation or not (output["call_record_clause"] and output["pid_process"] and output["udcp_process"]):
        output["total_points"] = "0"
    else:
        total_points = (
            float(output["call_opening_points"]) +
            float(output["customer_experience_points"]) +
            float(output["negotiation_points"]) +
            float(output["process_compliance_points"]) +
            float(output["documentation_points"]) +
            float(output["call_closing_points"]) +
            float(output["service_compliance_points"])
        )
        output["total_points"] = str(total_points)

    return output

if __name__ == "__main__":
    transcription = input_data
    result = evaluate_criteria(transcription)
    print(json.dumps(result, indent=4))

    with open('output.json', 'w') as outfile:
        json.dump(result, outfile, indent=4)