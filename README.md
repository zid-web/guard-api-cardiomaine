# Guard API — correctif extraction PDF (JSON malformé)

Ce dossier contient le correctif à déployer sur le repo
[`zid-web/guard-api-cardiomaine`](https://github.com/zid-web/guard-api-cardiomaine)
(Render : `guard-api-cardiomaine.onrender.com`).

## Symptôme

Import PDF côté planning :

`Réponse d'extraction invalide (JSON malformé) : Expecting value: line …`

Cause : Claude Vision renvoyait un JSON tronqué (`max_tokens=4000`) ou légèrement
invalide, et `json.loads` échouait sans récupération.

## Contenu

| Fichier | Rôle |
|---------|------|
| `llm_json.py` | Parser / réparation JSON LLM (nouveau) |
| `test_llm_json.py` | Tests unitaires |
| `pdf_upload.py` | `max_tokens=16000` + parse robuste + retry / repair |
| `pdf_vision_parser.py` | Utilise `parse_llm_json` |
| `voice_command.py` | Idem pour la voix |
| `main.py` | Accepte PDF même si MIME `octet-stream` |

Patch équivalent : `../patches/fix-pdf-json-extraction.patch`

## Déploiement

```bash
cd /path/to/guard-api-cardiomaine
git checkout -b cursor/fix-pdf-json-extraction-b101
cp /path/to/planning-cardiomaine/guard-api/{llm_json.py,test_llm_json.py,pdf_upload.py,pdf_vision_parser.py,voice_command.py,main.py} .
python3 -m unittest test_llm_json.py -v
git add llm_json.py test_llm_json.py pdf_upload.py pdf_vision_parser.py voice_command.py main.py
git commit -m "fix(pdf): robust LLM JSON parsing for planning extraction"
git push -u origin HEAD
# merger sur main → Render redéploie
```

Ou : `git am ../planning-cardiomaine/patches/fix-pdf-json-extraction.patch`
