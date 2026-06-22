```mermaid
graph TB
    subgraph "User Interface"
        UI[Streamlit Dashboard]
        API[FastAPI Backend]
        CLI[CLI / Scripts]
    end

    subgraph "Appliance Detection (Phase 1 Upgrade)"
        AD[Appliance Detector<br/>YOLO11s<br/>← fallback YOLOv8n]
        AD_CROI[Crop ROI]
    end

    subgraph "Damage Detection"
        DD[Damage Detector<br/>YOLO (bbox)<br/>Phone/Laptop/Fridge]
        DS[Damage Segmentation<br/>YOLO11s-seg<br/>← Phase 2 New]
    end

    subgraph "Analysis Services"
        MP[Missing Part Detector<br/>Rule-based]
        FD[Fraud Detection<br/>ELA + Metadata]
        SS[Severity Service<br/>% area → 4-tier]
        FS[Fraud Service<br/>7-factor engine]
        RS[Repair Service<br/>YAML cost config]
    end

    subgraph "Risk & Claims"
        CS[Claim Risk Engine<br/>Biz risk 0-100]
        CH[Claim History<br/>SQLite CRUD]
        PS[PDF Service<br/>Professional reports]
    end

    subgraph "Data"
        DB[(SQLite<br/>claim_history.db)]
        YAML[repair_costs.yaml]
        REPORTS[PDF Reports]
        MODELS[YOLO .pt Weights]
    end

    UI --> API
    CLI --> API
    API --> AD
    AD --> AD_CROI
    AD_CROI --> DD
    AD_CROI --> DS
    API --> MP
    API --> FD

    DD --> SS
    DS --> SS
    FD --> FS
    SS --> RS
    DS --> RS
    SS --> CS
    FS --> CS
    MP --> CS
    CS --> CH
    CH --> DB
    PS --> REPORTS
    RS --> YAML
    AD --> MODELS
    DD --> MODELS
    DS --> MODELS

    subgraph "Video Pipeline"
        VP[Frame Extraction]
        VF[Per-frame Pipeline]
        VO[Video Writer]
    end

    API --> VP --> VF --> VO

    subgraph "Demo Generator (Phase 10)"
        DG[Demo Generator<br/>video.mp4 → demo_video.mp4]
        DG --> VP
        VF --> DG
    end
```
