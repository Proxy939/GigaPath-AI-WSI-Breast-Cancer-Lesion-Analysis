# Checkpoints Directory

This directory is for **trained model weights**.

## Usage

1. Place your trained model checkpoint here:
   ```
   checkpoints/best_model.pth
   ```

2. **DO NOT commit model files to Git**
   - Model weights are ignored by `.gitignore`
   - Share models separately (Google Drive, S3, etc.)

## Expected Files

- `best_model.pth` - Production MIL model checkpoint
- `last_model.pth` - Last training epoch (optional)

## For Backend Developers

After cloning this repository:
1. Obtain `best_model.pth` from the ML team
2. Place it in this directory
3. Run inference: `python scripts/infer_mil.py --model checkpoints/best_model.pth`

---

**Note**: This directory must exist for scripts to function correctly.
