import json
import os
import jiwer
import sacrebleu
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import unicodedata

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text, for_translation=False):
    """Preprocess text for WER/CER or translation metrics."""
    text = unicodedata.normalize('NFKC', text.lower())
    if for_translation:
        return text  # Minimal normalization for translation metrics
    else:
        text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in '.,!?')
    return ' '.join(text.split())

def compute_der(gt_speakers, pred_speakers, duration, debug_log=None):
    """Compute diarization metrics: DER, MSR, FAR, SMR using a timeline-based approach."""
    gt_speakers = sorted([(spk.strip(), round(start, 3), round(end, 3)) for spk, start, end in gt_speakers], key=lambda x: x[1])
    pred_speakers = sorted([(spk.strip(), round(start, 3), round(end, 3)) for spk, start, end in pred_speakers], key=lambda x: x[1])
    
    if debug_log:
        debug_log.write(f"Ground Truth Speakers (raw): {[(spk, start, end) for spk, start, end in gt_speakers]}\n")
        debug_log.write(f"Predicted Speakers (raw): {[(spk, start, end) for spk, start, end in pred_speakers]}\n")
    
    # Initialize timeline with 1ms resolution and boolean dtype
    resolution = 1000  # 1000 units per second for higher precision
    timeline_size = int(duration * resolution) + 1
    gt_timeline = np.zeros(timeline_size, dtype=bool)
    pred_timeline = np.zeros(timeline_size, dtype=bool)
    smr_timeline = np.zeros(timeline_size, dtype=bool)
    
    # Fill ground truth timeline
    for spk, start, end in gt_speakers:
        start_idx = int(start * resolution)
        end_idx = min(int(end * resolution), timeline_size)
        gt_timeline[start_idx:end_idx] = True
    
    # Fill predicted timeline
    for spk, start, end in pred_speakers:
        start_idx = int(start * resolution)
        end_idx = min(int(end * resolution), timeline_size)
        pred_timeline[start_idx:end_idx] = True
    
    # Compute SMR by checking speaker mismatches in overlapping regions
    for i, (gt_spk, gt_start, gt_end) in enumerate(gt_speakers):
        for pred_spk, pred_start, pred_end in pred_speakers:
            if gt_spk != pred_spk:
                overlap_start = max(gt_start, pred_start)
                overlap_end = min(gt_end, pred_end)
                if overlap_start < overlap_end:
                    start_idx = int(overlap_start * resolution)
                    end_idx = min(int(overlap_end * resolution), timeline_size)
                    smr_timeline[start_idx:end_idx] = True
                    if debug_log:
                        debug_log.write(f"SMR: GT '{gt_spk}' ({gt_start}–{gt_end}) vs Pred '{pred_spk}' ({pred_start}–{pred_end}), Overlap {overlap_end-overlap_start}s\n")
    
    # Calculate errors
    missed_speech = np.sum(gt_timeline & ~pred_timeline) / resolution
    false_alarm = np.sum(pred_timeline & ~gt_timeline) / resolution
    speaker_mismatch = np.sum(smr_timeline) / resolution
    
    msr = missed_speech / duration if duration > 0 else 0.0
    far = false_alarm / duration if duration > 0 else 0.0
    smr = speaker_mismatch / duration if duration > 0 else 0.0
    der = msr + far + smr
    
    if debug_log and (msr > 0 or far > 0 or smr > 0):
        debug_log.write(f"DER={der:.2%}, MSR={msr:.2%}, FAR={far:.2%}, SMR={smr:.2%}, Duration={duration}s\n")
    
    return {
        "der": der,
        "msr": msr,
        "far": far,
        "smr": smr,
        "missed_speech_time": missed_speech,
        "false_alarm_time": false_alarm,
        "speaker_mismatch_time": speaker_mismatch,
        "duration": duration
    }

def evaluate_file(gt_file, pred_file, error_log):
    """Evaluate a single ground truth and prediction file pair."""
    errors = []
    
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
    except json.JSONDecodeError as e:
        error_log.write(f"Invalid JSON in ground truth file {gt_file}: {e}\n\n")
        return None
    
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
    except json.JSONDecodeError as e:
        error_log.write(f"Invalid JSON in prediction file {pred_file}: {e}\n\n")
        return None
    
    # Extract duration from ground truth JSON
    try:
        duration = float(gt_data.get("call_duration", "0s").rstrip('s'))
        if duration <= 0:
            raise ValueError("Invalid or missing call_duration")
    except (ValueError, TypeError) as e:
        error_log.write(f"Error parsing call_duration in {gt_file}: {e}\n")
        # Fallback to max timestamp
        try:
            duration = max(seg["end"] for seg in gt_data["transcription"] + pred_data["transcription"])
        except (KeyError, ValueError) as e:
            error_log.write(f"Error computing fallback duration for {gt_file}: {e}\n")
            return None
    
    try:
        gt_trans = " ".join(seg["text"] for seg in gt_data["transcription"])
        pred_trans = " ".join(seg["text"] for seg in pred_data["transcription"])
        gt_trans_norm = preprocess_text(gt_trans)
        pred_trans_norm = preprocess_text(pred_trans)
        
        if gt_trans_norm == pred_trans_norm:
            wer = 0.0
            cer = 0.0
            errors.append("Transcriptions identical: No errors detected")
        else:
            measures = jiwer.compute_measures(gt_trans_norm, pred_trans_norm)
            wer = measures['wer']
            cer = jiwer.cer(gt_trans_norm, pred_trans_norm)
            errors.append(f"Transcription errors detected: WER={wer:.2%}, CER={cer:.2%}")
    except KeyError as e:
        error_msg = f"Missing key in transcription data for {gt_file}: {e}"
        errors.append(error_msg)
        error_log.write(error_msg + "\n")
        wer = cer = 1.0
    
    try:
        gt_speakers = [(seg["speaker"], seg["start"], seg["end"]) for seg in gt_data["transcription"]]
        pred_speakers = [(seg["speaker"], seg["start"], seg["end"]) for seg in pred_data["transcription"]]
        correct_channels = 0
        min_segments = min(len(gt_speakers), len(pred_speakers))
        for gt, pred in zip(gt_speakers[:min_segments], pred_speakers[:min_segments]):
            gt_spk, gt_start, gt_end = gt
            pred_spk, pred_start, pred_end = pred
            if gt_spk.strip() == pred_spk.strip() and abs(gt_start - pred_start) < 1.0 and abs(gt_end - pred_end) < 1.0:
                correct_channels += 1
        channel_accuracy = correct_channels / len(gt_speakers) if gt_speakers else 1.0
        if len(gt_speakers) != len(pred_speakers):
            errors.append(f"Speaker segment count mismatch: GT {len(gt_speakers)}, Pred {len(pred_speakers)}")
        if channel_accuracy < 1.0:
            errors.append(f"Channel accuracy: {channel_accuracy:.2%}")
    except KeyError as e:
        error_msg = f"Missing key in speaker data for {gt_file}: {e}"
        errors.append(error_msg)
        error_log.write(error_msg + "\n")
        channel_accuracy = 0.0
    
    try:
        diarization_results = compute_der(gt_speakers, pred_speakers, duration, error_log)
        der = diarization_results["der"]
        msr = diarization_results["msr"]
        far = diarization_results["far"]
        smr = diarization_results["smr"]
        
        if der > 0.15:
            errors.append(f"High DER: {der:.2%} (MSR={msr:.2%}, FAR={far:.2%}, SMR={smr:.2%})")
        elif any(rate > 0.15 for rate in [msr, far, smr]):
            errors.append(f"Diarization issues: MSR={msr:.2%}, FAR={far:.2%}, SMR={smr:.2%}")
    except (KeyError, ValueError) as e:
        error_msg = f"Error computing diarization metrics for {gt_file}: {e}"
        errors.append(error_msg)
        error_log.write(error_msg + "\n")
        der = msr = far = smr = 1.0
    
    try:
        gt_translations = [pair["english"] for pair in gt_data["translation"]]
        pred_translations = [pair["english"] for pair in pred_data["translation"]]
        gt_translations_norm = [preprocess_text(t, for_translation=True) for t in gt_translations]
        pred_translations_norm = [preprocess_text(t, for_translation=True) for t in pred_translations]
        
        # Debug logging for translation pairs
        if len(gt_translations) != len(pred_translations):
            error_log.write(f"Translation count mismatch in {gt_file}: GT {len(gt_translations)}, Pred {len(pred_translations)}\n")
        for i, (gt_t, pred_t) in enumerate(zip(gt_translations_norm, pred_translations_norm)):
            if gt_t != pred_t:
                error_log.write(f"Translation mismatch in {gt_file} at index {i}: GT '{gt_t}' vs Pred '{pred_t}'\n")
        
        bleu = sacrebleu.corpus_bleu(pred_translations_norm, [gt_translations_norm], tokenize='none').score / 100
        meteor_scores = [
            meteor_score([word_tokenize(gt)], word_tokenize(pred))
            for gt, pred in zip(gt_translations_norm, pred_translations_norm)
            if gt and pred
        ]
        meteor = np.mean(meteor_scores) if meteor_scores else 0.0
        chrf = sacrebleu.corpus_chrf(pred_translations_norm, [gt_translations_norm]).score / 100
        
        if bleu < 0.3:
            errors.append(f"Low BLEU score: {bleu:.4f}")
    except Exception as e:
        error_msg = f"Error computing translation metrics for {gt_file}: {e}"
        errors.append(error_msg)
        error_log.write(error_msg + "\n")
        bleu = meteor = chrf = 0.0
    
    return {
        "file": os.path.basename(gt_file),
        "wer": wer,
        "cer": cer,
        "channel_accuracy": channel_accuracy,
        "der": der,
        "msr": msr,
        "far": far,
        "smr": smr,
        "bleu": bleu,
        "meteor": meteor,
        "chrf": chrf,
        "errors": errors
    }

def generate_report(results, output_pdf):
    """Generate a PDF report summarizing evaluation results."""
    try:
        doc = SimpleDocTemplate(output_pdf, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        elements.append(Paragraph("Model Evaluation Report", styles['Title']))
        elements.append(Spacer(0, 72))
        
        data = [
            ["Metric", "Average"],
            ["WER", f"{np.mean([r['wer'] for r in results]):.2%}"],
            ["CER", f"{np.mean([r['cer'] for r in results]):.2%}"],
            ["Channel Accuracy", f"{np.mean([r['channel_accuracy'] for r in results]):.2%}"],
            ["DER", f"{np.mean([r['der'] for r in results]):.2%}"],
            ["MSR", f"{np.mean([r['msr'] for r in results]):.2%}"],
            ["FAR", f"{np.mean([r['far'] for r in results]):.2%}"],
            ["SMR", f"{np.mean([r['smr'] for r in results]):.2%}"],
            ["BLEU", f"{np.mean([r['bleu'] for r in results]):.4f}"],
            ["METEOR", f"{np.mean([r['meteor'] for r in results]):.4f}"],
            ["chrF", f"{np.mean([r['chrf'] for r in results]):.4f}"]
        ]
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(0, 12))
        
        elements.append(Paragraph("Per-File Error Log", styles['Heading2']))
        for result in results:
            if result["errors"]:
                elements.append(Paragraph(f"File: {result['file']}", styles['Heading3']))
                for error in result["errors"]:
                    elements.append(Paragraph(error, styles['Normal']))
                elements.append(Spacer(0, 6))
        
        doc.build(elements)
    except Exception as e:
        print(f"Error generating PDF report: {e}")

def main():
    """Main function to process all files and generate report."""
    gt_dir = "Ground Truth/Singlish calls"
    pred_dir = "Model Prediction/Singlish calls"
    output_pdf = "evaluation_report.pdf"
    error_log_file = "error_logs.txt"
    
    results = []
    
    with open(error_log_file, 'w', encoding='utf-8') as log_f:
        log_f.write(f"Processing {len([f for f in os.listdir(gt_dir) if f.endswith('.json')])} ground truth JSON files\n")
        for gt_filename in sorted(os.listdir(gt_dir)):
            if gt_filename.endswith('.json'):
                gt_path = os.path.join(gt_dir, gt_filename)
                pred_path = os.path.join(pred_dir, gt_filename)
                
                log_f.write(f"\nAttempting to process {gt_filename}\n")
                if os.path.exists(pred_path):
                    result = evaluate_file(gt_path, pred_path, log_f)
                    if result:
                        results.append(result)
                    
                    if result:
                        log_f.write(f"File: {result['file']}\n")
                        log_f.write(f"WER: {result['wer']:.2%}\n")
                        log_f.write(f"CER: {result['cer']:.2%}\n")
                        log_f.write(f"Channel Accuracy: {result['channel_accuracy']:.2%}\n")
                        log_f.write(f"DER: {result['der']:.2%} (MSR={result['msr']:.2%}, FAR={result['far']:.2%}, SMR={result['smr']:.2%})\n")
                        log_f.write(f"BLEU: {result['bleu']:.4f}\n")
                        log_f.write(f"METEOR: {result['meteor']:.4f}\n")
                        log_f.write(f"chrF: {result['chrf']:.4f}\n")
                        if result["errors"]:
                            log_f.write("Errors:\n")
                            for error in result["errors"]:
                                log_f.write(f"- {error}\n")
                        log_f.write("\n")
                else:
                    log_f.write(f"Prediction file not found for {gt_filename}\n\n")
    
    if results:
        generate_report(results, output_pdf)
        print(f"Evaluation report generated: {output_pdf}")
        print(f"Error log saved: {error_log_file}")
        print(f"Processed {len(results)} files")
    else:
        print("No valid files processed.")

if __name__ == "__main__":
    main()