import json

def check_escalation_from_file(file_path: str, report_type: str) -> tuple[str, str]:
    """
    Check if a Pre-Due or Post-Due report (from JSON file) is escalation worthy.
    Args:
        file_path (str): Path to JSON deviance report.
        report_type (str): "predue" or "postdue"
    Returns:
        tuple[str, str]: ("Escalation" or "No Escalation", failing criterion or empty string)
    """
    # Load report from file
    with open(file_path, "r") as f:
        report = json.load(f)
    # Escalation-worthy criteria (strict check, no scoring)
    escalation_criteria = {
        "predue": [
            "verification_security",  # Verification for Account Security
            "follow_policies_procedure"  # Follow Policies Procedure
        ],
        "postdue": [
            "verification_security",  # Verification for Account Security
            "follow_policies_procedure",  # Follow Policies Procedure
            "state_reason_of_call",  # State Reason of Call
            "probing_questions_effectiveness",
            "payment_resolution_actions",
            "payment_delay_consequences"  # Negotiation Strategy
        ]
    }
    # Normalize report_type
    report_type = report_type.lower()
    if report_type not in escalation_criteria:
        raise ValueError("report_type must be 'predue' or 'postdue'")
    # Strict escalation check
    for key in escalation_criteria[report_type]:
        if key not in report:
            raise KeyError(f"Missing key '{key}' in {report_type} report")
        if report[key] is False:
            return "Escalation", f"{key} criteria acts as outlier"
    return "No Escalation", ""

def main():
    # Assume deviance_report.json exists in the same directory
    file_path = "deviance_report.json"
    
    # Load JSON to get report_type
    try:
        with open(file_path, "r") as f:
            report = json.load(f)
        
        # Extract and normalize report_type from call_type
        if "call_type" not in report:
            raise KeyError("Missing 'call_type' in JSON report")
        call_type = report["call_type"]
        report_type = call_type.lower()
        if report_type == "pre-due":
            report_type = "predue"
        elif report_type == "post-due":
            report_type = "postdue"
        else:
            raise ValueError(f"Invalid call_type '{call_type}' in JSON report. Must_responses be 'Pre-Due' or 'Post-Due'")
        
        # Run escalation check
        result, failing_criterion = check_escalation_from_file(file_path, report_type)
        if result == "Escalation":
            print(f"Deviance Report Result for {call_type} call: {result}, {failing_criterion}")
        else:
            print(f"Deviance Report Result for {call_type} call: {result}")
    except (KeyError, ValueError, FileNotFoundError) as e:
        print("Error:", str(e))

if __name__ == "__main__":
    main()