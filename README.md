# Chatbot pour mon projet de Bachelor en Sciences informatiques

Chatbot fait à l'aide de Tensorflow 1.0. Le code ne fonctionne pas pour des versions ultérieures.

## Dépendances

* tensorflow
* numpy
* six
* clint

## Lancement du code

Tout d'abord avant de lancer le code il faut télécharger le data set en cliquant [ici](http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/ubuntu_dialogs.tgz). Ensuite il faut décomprésser l'archive dans le dossier /data. Pour lancer le code avec les valeurs par défaut de mon modèle il suffit de lancer les commandes ci-dessous dans l'ordre.

* Entrainer l'algorithme

```
python3 exec.py
```

* Tester le modèle (pas obligatoire si vous voulez juste parler avec le bot):

```
python3 exec.py --self_test
```

* Parler avec le chatbot:

```
python3 exec.py --decode
```

## Options du code

On peut changer les valeurs par défaut du code en mettant en option au lancement: `--{nom_variable}={val}`.

**Attention**, il faut impérativement lancer l'entrainement, le test et le décodage avec les **mêmes** paramètres.

#### Les options principales

Option | Valeur par défaut | Utilité
--- | --- | ---
**size** | 1024 | Nombre de cellules d'une couche du modèle
**num_layers** | 1 | Nombre de couches du modèle 
**num_attns** | 1 | Nombre de vecteurs d'attentions pour le décodage 
**num_attns_output** | 0 | Nombre de vecteurs d'attentions appliqués sur la sortie générée
**from_vocab_size** | 40000 | Taille maximale du vocabulaire d'entrée
**to_vocab_size** | 40000 | Taille maximale du vocabulaire de sortie
**steps_per_checkpoint** | 5 | Nombre d'étapes avant de sauvegarder l'état du modèle

Par exemple pour entrainer le modèle avec 3 couches de 256 cellules:
```
python3 exec.py --size=256 --num_layers=3
```
et pour lancer le décodage:
```
python3 exec.py --decode --size=256 --num_layers=3
``` 
Il y a  d'autres options qui sont au début du fichier `exec.py`.