**Virtual training coach 
Powered by MediaPipe**

_Fernando Chirici
7051275
Esame di Image and video Analisys 
Professore: Pietro Pala_
_________________________________________________

**Requirements**

1. Python >= 3.7

**Istruzioni per l'utilizzo**

1. **Preparazione all'uso**

   1. Da terminale, navigare fino alla directory del progetto
   `cd C:\Users\...\IVA_fernando_chirici`
   2. Creare un python virtual environment chiamato VTC (Virtual Training Coach)
   `python -m venv VTC`
   3. Attivare l'environment appena creato
   `VTC\Scripts\activate`
   4. Installare i requirements
   `pip install -r requirements.txt`



2. **Avvio del virtual trainer coach (VTC)**

**NOTE:** 
Il codice si suddivide in tre parti: 
1. Pre processing, 
2. Skeleton extraction
3. Post processing

-> La configurazione avviene interamente dal file di configurazione _config.json_. 
   Prima di poter effettuare l'analisi della postura di un qualsiasi esercizio fisico è necessario avviare il VTC
   in modalità "_trainer_", settando la valore "trainer" = "true", ed inserendo nel _video_path_, il video di un 
   esercizio svolto in modo corretto. 
   
-> Una volta terminata l'esecuzione sarà possibile effettuare l'analisi inserendo, come prima,
   in _video_path_, il path del video che si vuole analizzare ed impostando il flag _trainer_ a "false".

   Per gli altri parametri segue una breve descrizione:
   -_exercise_: nome dell'esercizio. Questo deve rimanere uguale sia per l'esecuzione del video di training che per quella 
   dedicata all'analisi.
   -_extraction_flag_: da porre a "false" se dal Pre processing, si vuole passare direttamente al Post processing.
   Questo si può fare se è già stata effettuata l'estrazione degli scheletri ad un avvio precedente configurato con
   extraction_flag = "true".
   -EUCLIDEAN / ANGULAR / COMBINED: da porre ad **1** il tipo di analisi che si vuole effettuare ed a **0** gli altri.
   E' possibile eseguire molteplici analisi in una sola esecuzione (settando ad 1 una o più tipologie).

 
-> Per comodità è consigliato posizionare dentro la cartella _uservideos_ tutti i video che si intende utilizzare.

-> Se il flag _trainer_ è true, i valori di EUCLIDEAN / ANGULAR / COMBINED, nel config.json, non influiscono sull'esecuzione.


**Avvio dello script**

1. Da terminale già posizionato nel folder del progetto, si avvia, **in seguito alla compilazione e salvataggio del file config.json** 
con un qualsiasi editor di testo, con `python main.py`


**Output**

L'output si può osservare sotto forma di video alla directory `IVA_fernando_chirici/videos/out/`
