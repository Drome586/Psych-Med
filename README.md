# PsychDialog Dataset

1.we construct a large-scale psychological counseling dialogue dataset, **PsychDialog**, collected from three Chinese online health community platforms. PsychDialog contains **3,268** multi-turn dialogues between doctors and patients together with a wide range of related metadata including **visited hospitals, visited departments, the types of disorders and patient self-reports,** which can advance the related research to build accurate and personalized dialogue models. 

2.We show some sample datasets for this purpose：

```json
{
    "id": 0,
    "Hospital": "北京中医药大学东直门医院",
    "Department": "心理门诊",
    "Disease": "焦虑症",
    "Situation": "因体检出现乙肝表面抗体阳性...",
    "Dialogue": [
        "医生：...",
        "病人：...",
        "医生：..."
    ],
    "Dia_turn": 2
}
```

# code availability

We provide all the code needed for the experiment and store the information to be noted during the experiment in the README.md file under the specific file.