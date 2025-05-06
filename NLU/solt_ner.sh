cd C:\Users\19981\Documents\GitHub\MedicalKB\NLU\Slot_filling
nohup gunicorn -w 1 -b 0.0.0.0:6002 solt_app:app > C:\Users\19981\Documents\GitHub\MedicalKB\NLU\logs/solt_output.log 2>&1 &
#python solt_app.py