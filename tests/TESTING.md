# Guide des Rapports de Tests

## Configuration automatique

Avec la configuration actuelle dans `pytest.ini`, chaque exécution de pytest génère automatiquement :

1. **Rapport HTML des tests** : `reports/report.html`
2. **Rapport de couverture HTML** : `reports/coverage/index.html`

## Utilisation

### Exécution simple

```bash
pytest
```

### Options supplémentaires disponibles

```bash
# Tests avec plus de détails
pytest -v

# Tests uniquement rapides (exclure les lents)
pytest -m "not slow"

# Tests d'une classe spécifique
pytest tests/test_snake_game.py::TestSnakeGame

# Tests avec couverture détaillée
pytest --cov-report=term-missing
```

## Fichiers générés

### `reports/report.html`

- **Résumé des tests** : Nombre de tests passés/échoués
- **Détails par test** : Temps d'exécution, status
- **Erreurs détaillées** : Stack traces complètes pour les échecs
- **Navigation facile** : Interface web interactive

### `reports/coverage/index.html`

- **Couverture globale** : Pourcentage de code testé
- **Couverture par fichier** : Détail module par module
- **Lignes manquantes** : Code non testé mis en évidence
- **Navigation dans le code** : Visualisation ligne par ligne

## Ouverture des rapports

### Windows

```bash
# Ouvrir le rapport de tests
start reports/report.html

# Ouvrir le rapport de couverture
start reports/coverage/index.html
```

### Depuis VS Code

- Clic droit sur le fichier → "Open with Live Server"
- Ou utiliser l'extension "Preview on Web Server"

## Markers disponibles

- `@pytest.mark.slow` : Tests lents (peuvent être exclus)
- `@pytest.mark.integration` : Tests d'intégration
- `@pytest.mark.unit` : Tests unitaires

## Exemple d'utilisation avec markers

```python
@pytest.mark.slow
def test_long_simulation():
    # Test qui prend du temps
    pass

@pytest.mark.unit
def test_quick_function():
    # Test rapide
    pass
```

## Configuration actuelle dans pytest.ini

```ini
addopts = "-ra -q --strict-markers --html=reports/report.html --self-contained-html --cov=src --cov-report=html:reports/coverage"
```

- `--html=reports/report.html` : Génère le rapport HTML
- `--self-contained-html` : Rapport autonome (CSS/JS inclus)
- `--cov=src` : Mesure la couverture du dossier src
- `--cov-report=html:reports/coverage` : Rapport de couverture HTML
