
import json
import os
import unicodedata
from typing import List, Tuple, Dict, Any

def normalize_text(text: str) -> str:
    """Normalize text for comparison (strip, lowercase, NFKC)."""
    return unicodedata.normalize('NFKC', text.strip().lower())

def validate_json_file(filepath: str) -> Tuple[bool, str]:
    """Check if a file is valid JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json.load(f)
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def check_structure(data: Dict[str, Any], filename: str) -> List[str]:
    """Verify JSON structure and data integrity."""
    errors = []
    
    # Check top-level keys
    if not isinstance(data, dict):
        return [f"Root is not a dictionary"]
    if "transcription" not in data or "translation" not in data:
        return [f"Missing 'transcription' or 'translation' in {filename}"]
    
    # Check transcription
    if not isinstance(data["transcription"], list):
        errors.append(f"'transcription' is not a list")
    else:
        for i, seg in enumerate(data["transcription"]):
            if not isinstance(seg, dict):
                errors.append(f"Transcription segment {i} is not a dictionary")
                continue
            for key in ["speaker", "start", "end", "text"]:
                if key not in seg:
                    errors.append(f"Segment {i} missing '{key}'")
                    continue
                if key == "speaker" and (not isinstance(seg[key], str) or not seg[key].strip()):
                    errors.append(f"Segment {i} 'speaker' is invalid: {seg[key]}")
                elif key == "text" and (not isinstance(seg[key], str) or not seg[key].strip()):
                    errors.append(f"Segment {i} 'text' is invalid: {seg[key]}")
                elif key in ["start", "end"] and not isinstance(seg[key], (int, float)):
                    errors.append(f"Segment {i} '{key}' is not numeric: {seg[key]}")
                elif key == "start" and seg[key] < 0:
                    errors.append(f"Segment {i} 'start' is negative: {seg[key]}")
                elif key == "end" and seg[key] <= seg["start"]:
                    errors.append(f"Segment {i} 'end' <= 'start': {seg[key]} <= {seg['start']}")
    
    # Check translation
    if not isinstance(data["translation"], list):
        errors.append(f"'translation' is not a list")
    else:
        for i, pair in enumerate(data["translation"]):
            if not isinstance(pair, dict):
                errors.append(f"Translation {i} is not a dictionary")
                continue
            if "english" not in pair:
                errors.append(f"Translation {i} missing 'english'")
                continue
            if not isinstance(pair["english"], str) or not pair["english"].strip():
                errors.append(f"Translation {i} 'english' is invalid: {pair['english']}")
    
    return errors

def compare_speakers(gt_data: Dict, pred_data: Dict, filename: str) -> List[str]:
    """Compare speaker labels and timestamps for SMR issues."""
    issues = []
    
    gt_segs = gt_data.get("transcription", [])
    pred_segs = pred_data.get("transcription", [])
    
    if len(gt_segs) != len(pred_segs):
        issues.append(f"Segment count mismatch: GT={len(gt_segs)}, Pred={len(pred_segs)}")
        return issues
    
    for i, (gt_seg, pred_seg) in enumerate(zip(gt_segs, pred_segs)):
        gt_speaker = normalize_text(gt_seg.get("speaker", ""))
        pred_speaker = normalize_text(pred_seg.get("speaker", ""))
        gt_start = round(float(gt_seg.get("start", 0)), 2)
        gt_end = round(float(gt_seg.get("end", 0)), 2)
        pred_start = round(float(pred_seg.get("start", 0)), 2)
        pred_end = round(float(pred_seg.get("end", 0)), 2)
        
        if gt_speaker != pred_speaker:
            issues.append(f"Segment {i} speaker mismatch: GT='{gt_seg['speaker']}' vs Pred='{pred_seg['speaker']}'")
        if abs(gt_start - pred_start) > 0.01:
            issues.append(f"Segment {i} start time mismatch: GT={gt_start} vs Pred={pred_start}")
        if abs(gt_end - pred_end) > 0.01:
            issues.append(f"Segment {i} end time mismatch: GT={gt_end} vs Pred={pred_end}")
    
    return issues

def check_directories(gt_dir: str, pred_dir: str) -> None:
    """Check file synchronization, JSON validity, and structure."""
    report_file = "file_check_report.txt"
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.json')])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.json')])
    
    with open(report_file, 'w', encoding='utf-8') as report:
        # 1. Check synchronization
        report.write("=== File Synchronization Check ===\n")
        missing_in_pred = [f for f in gt_files if f not in pred_files]
        missing_in_gt = [f for f in pred_files if f not in gt_files]
        common_files = [f for f in gt_files if f in pred_files]
        
        report.write(f"Ground Truth Files: {len(gt_files)}\n")
        report.write(f"Model Prediction Files: {len(pred_files)}\n")
        report.write(f"Common Files: {len(common_files)}\n")
        if missing_in_pred:
            report.write("Files missing in Model Prediction:\n")
            for f in missing_in_pred:
                report.write(f"- {f}\n")
        if missing_in_gt:
            report.write("Files missing in Ground Truth:\n")
            for f in missing_in_gt:
                report.write(f"- {f}\n")
        report.write("\n")
        
        # 2. Validate JSON and structure
        report.write("=== JSON Validity and Structure Check ===\n")
        invalid_files = []
        structural_issues = []
        smr_issues = []
        
        for filename in common_files:
            gt_path = os.path.join(gt_dir, filename)
            pred_path = os.path.join(pred_dir, filename)
            
            report.write(f"\nFile: {filename}\n")
            
            # Ground Truth
            gt_valid, gt_error = validate_json_file(gt_path)
            if not gt_valid:
                report.write(f"Ground Truth: {gt_error}\n")
                invalid_files.append(filename)
                continue
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            gt_struct_errors = check_structure(gt_data, f"Ground Truth/{filename}")
            if gt_struct_errors:
                report.write("Ground Truth Structure Errors:\n")
                for err in gt_struct_errors:
                    report.write(f"- {err}\n")
                structural_issues.append((filename, "Ground Truth", gt_struct_errors))
            
            # Model Prediction
            pred_valid, pred_error = validate_json_file(pred_path)
            if not pred_valid:
                report.write(f"Model Prediction: {pred_error}\n")
                invalid_files.append(filename)
                continue
            with open(pred_path, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)
            pred_struct_errors = check_structure(pred_data, f"Model Prediction/{filename}")
            if pred_struct_errors:
                report.write("Model Prediction Structure Errors:\n")
                for err in pred_struct_errors:
                    report.write(f"- {err}\n")
                structural_issues.append((filename, "Model Prediction", pred_struct_errors))
            
            # Compare speakers for SMR
            smr_errors = compare_speakers(gt_data, pred_data, filename)
            if smr_errors:
                report.write("SMR-Related Issues:\n")
                for err in smr_errors:
                    report.write(f"- {err}\n")
                smr_issues.append((filename, smr_errors))
        
        # 3. Summary
        report.write("\n=== Summary ===\n")
        report.write(f"Total Files Checked: {len(common_files)}\n")
        report.write(f"Invalid JSON Files: {len(invalid_files)}\n")
        report.write(f"Files with Structural Issues: {len(structural_issues)}\n")
        report.write(f"Files with SMR Issues: {len(smr_issues)}\n")
        if invalid_files:
            report.write("Invalid JSON Files List:\n")
            for f in invalid_files:
                report.write(f"- {f}\n")
        if structural_issues:
            report.write("Structural Issues List:\n")
            for f, dir_type, errors in structural_issues:
                report.write(f"- {dir_type}/{f}: {len(errors)} issues\n")
        if smr_issues:
            report.write("SMR Issues List:\n")
            for f, errors in smr_issues:
                report.write(f"- {f}: {len(errors)} issues\n")
    
    print(f"Check complete. Report saved to {report_file}")
    print(f"Ground Truth Files: {len(gt_files)}")
    print(f"Model Prediction Files: {len(pred_files)}")
    print(f"Common Files: {len(common_files)}")
    print(f"Invalid JSON Files: {len(invalid_files)}")
    print(f"Structural Issues: {len(structural_issues)}")
    print(f"SMR Issues: {len(smr_issues)}")

if __name__ == "__main__":
    gt_dir = "Ground Truth"
    pred_dir = "Model Prediction"
    check_directories(gt_dir, pred_dir)