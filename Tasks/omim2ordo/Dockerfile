# Étape 1 : Image de base Python 3.11.10
FROM python:3.11.10-slim

# Étape 2 : Installer les dépendances système et Java
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk && \
    apt-get clean


# Configurer JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Étape 3 : Définir le répertoire de travail
WORKDIR /app

# Étape 4 : Copier les scripts et les dépendances
COPY omim2ordo.py /app/
COPY ../../requirements.txt /app/
COPY download_data.py /app/

# Étape 5 : Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Étape 6 : Télécharger et organiser les données
RUN python download_data.py

# Étape 7 : Définir les volumes pour les résultats
VOLUME ["/results"]

# Étape 8 : Configurer la mémoire JVM par défaut
ENV JVM_MEMORY=8g

# Étape 9 : Commande d'exécution
CMD ["python", "omim2ordo.py"]
