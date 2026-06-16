import json
from typing import Dict, List, Optional, Any

from openai import OpenAI
from gmft_pymupdf import PyMuPDFDocument
from gmft.formatters.page.embed import embed_tables
from gmft.auto import AutoTableDetector, AutoTableFormatter

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from extractor.models import (
    ResearchData,
    Variables,
    StatisticalTest,
)

FEW_SHOT = True


class StructuredExtractor:
    """Extract structured data from research publications using a multi-step approach"""

    def __init__(
        self,
        api_key: str,
        dag: Optional[Any] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.model = model
        self.dag = dag
        self.client = OpenAI(api_key=api_key)
        self.extraction_steps = [
            self._extract_study_info,
            self._extract_variables,
            self._extract_statistics,
            self._refine_variables,
            self._extract_tests,
        ]
        self.extraction_history = []

    def _extract_texts(self, pdf_path: str) -> str:
        # Load PDF document with PyMuPDF for better line break detection
        doc = PyMuPDFDocument(pdf_path)
        # Initialize detector and formatter
        detector = AutoTableDetector()
        formatter = AutoTableFormatter()

        # Extract all tables from all pages
        all_tables = []
        for page in doc:
            # Detect tables on the page
            tables = detector.extract(page)
            # Format detected tables
            formatted_tables = [formatter.extract(table) for table in tables]
            all_tables.extend(formatted_tables)

        # Embed tables directly into document text as markdown
        formatted_pages = embed_tables(doc, tables=all_tables)

        all_text = []
        for page in formatted_pages:
            all_text.append(page.get_text())

        # Join all pages into one document
        content = "\n\n".join(all_text)

        # save the content to a markdown file where pdf_path is
        markdown_path = pdf_path.replace(".pdf", ".md")
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Close the document
        doc.close()
        return content

    def extract(self, pdf_path: str) -> ResearchData:
        """Main pipeline to extract structured data from PDF"""
        # Initialize empty research data
        research_data = ResearchData()

        content = self._extract_texts(pdf_path)
        print(content[:500])
        # Step by step extraction
        for step_func in self.extraction_steps:
            research_data = step_func(content, research_data)
            self.extraction_history.append(
                {
                    "step": step_func.__name__,
                    "data": research_data.model_dump(),
                }
            )

        return research_data

    @retry(
        retry=retry_if_exception_type((ValueError, json.JSONDecodeError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def _llm_extract(
        self,
        content: str,
        system_prompt: str,
        existing_data: Dict = None,
        examples: List[Dict] = None,
    ) -> Dict:
        """Make LLM API call with retry logic and few-shot examples"""

        messages = [{"role": "system", "content": system_prompt}]
        # Add examples if provided
        if examples and FEW_SHOT:
            for example in examples:
                messages.append({"role": "user", "content": example["input"]})
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(example["output"], indent=2),
                    }
                )

        user_content = f"Document content:\n{content}\n\n"
        if existing_data:
            user_content += f"Previously extracted data:\n{json.dumps(existing_data, indent=2)}\n\n"

        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0.0,
        )

        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            raise ValueError("Failed to parse JSON response")

    def _extract_study_info(
        self, content: str, data: ResearchData
    ) -> ResearchData:
        """Extract study size, groups, and group sizes"""
        system_prompt = """
        You are a specialized statistical extractor focusing ONLY on identifying core study parameters. Focus on tables, since they dictate the structure of the data. The groups are often defined as column headers in tables or explicitly mentioned in the text.
        Extract the following information from the research paper:
        1. study_size (total number of participants)
        2. groups (e.g., "Drug intervention group", "Control group", etc.)
        3. group_sizes (number of participants in each group)
        
        Return the data in the following JSON format:
        {
            "study_size": int,
            "groups": [string, string, ...],
            "group_sizes": {"group_name": int, ...}
        }

        When extracting group information, always use the most granular (detailed) subgroup definitions available from the text or tables. If a value (e.g., group_size) is only reported for a broader (meta) group, assign that value to all its subgroups. Ensure that the sum of all group_sizes equals the study_size. If any information is missing, use None. Aim for short and concise group names. If the group name is too long, use an abbreviation or a short form.

        ---
        Preview of the next step:
        In the next step, you will be asked to extract all variables measured in the study and categorize them by type (continuous, ordinal, categorical, binary), using the group names and sizes you provide here.
        """

        example_1 = {
            "input": """
        | Characteristic | Treatment A (n=45) | Treatment B (n=38) | Control (n=42) |
        |----------------|-------------------|-------------------|----------------|
        | Age, mean (SD) | 64.2 (12.1)      | 62.8 (11.9)      | 63.1 (12.4)   |
        | BMI, kg/m²     | 28.1 ± 4.2        | 27.9 ± 3.8        | 28.5 ± 4.1     |
        """,
            "output": {
                "study_size": 125,
                "groups": ["Treatment A", "Treatment B", "Control"],
                "group_sizes": {
                    "Treatment A": 45,
                    "Treatment B": 38,
                    "Control": 42,
                },
            },
        }
        example_2 = {
            "input": """
        A total of 180 participants were enrolled in this randomized controlled trial. 
        Participants were randomly assigned to either the intervention group (n=90) 
        receiving the new drug treatment or the control group (n=90) receiving placebo.
        """,
            "output": {
                "study_size": 180,
                "groups": ["Intervention", "Control"],
                "group_sizes": {"Intervention": 90, "Control": 90},
            },
        }
        example_3 = {
            "input": """
        | Demographics | Low Dose (n=32) | High Dose (n=35) | Placebo (n=33) | Total (N=100) |
        |--------------|-----------------|------------------|----------------|---------------|
        | Age (years)  | 58.3 ± 12.4     | 59.1 ± 11.8      | 57.9 ± 13.2    | 58.4 ± 12.5   |
        """,
            "output": {
                "study_size": 100,
                "groups": ["Low Dose", "High Dose", "Placebo"],
                "group_sizes": {
                    "Low Dose": 32,
                    "High Dose": 35,
                    "Placebo": 33,
                },
            },
        }

        examples = [example_1, example_2, example_3]

        result = self._llm_extract(content, system_prompt, examples=examples)

        # Use model_construct to update only specific fields
        updated_data = ResearchData.model_construct(
            _fields_set={"study_size", "groups", "group_sizes"},
            study_size=result.get("study_size"),
            groups=result.get("groups", []),
            group_sizes=result.get("group_sizes", {}),
            variables=data.variables,
            statistical_tests=data.statistical_tests,
        )

        return updated_data

    def _extract_variables(
        self, content: str, data: ResearchData
    ) -> ResearchData:
        """Identify and categorize variables"""
        target_variables = []
        if self.dag is not None:
            target_variables = list(self.dag.nodes())

        # Build strict DAG guidance
        dag_guidance = ""
        if target_variables:
            dag_guidance = f"""
            REQUIRED VARIABLES (extract ONLY these, ignore all others):
            {', '.join(target_variables)}
            CRITICAL INSTRUCTIONS:
            - Extract ONLY variables that match the names in the list above
            - Use the EXACT names provided in the list
            - If a measurement in the paper seems related but has a different name, map it to the closest matching variable from the list
            - DO NOT extract any variables not in this list
            - If you cannot find a measurement for a listed variable, omit it from the output (do not guess or infer)
        """

        system_prompt = f"""
        You are a specialized statistical extractor focusing ONLY on identifying specific variables.
        Based on the previously extracted study parameters, extract the requested variables and categorize them by type.

        {dag_guidance}

        Categorize each variable as:

        1. Continuous variables (e.g., Weight, Height, BMI): 
        - Variables measured on a numerical, truly continuous scale
        - Can theoretically take any value within a range (including decimals)

        2. Ordinal variables (e.g., Age, Ratings, Grade):
        - Variables measured on a numerical, continuous scale but can only take discrete values
        - Can only take whole numbers (no decimals)
        - Can be interpreted as a rounded version of a continuous variable

        RULE: In continuous and ordinal variables, distributions are described by mean, std, median, IQR, ... and there are no counts. Counts only occur in binary and categorical variables. Thus, if there is a count, it is either binary or categorical and if there is a mean, std, median, IQR, ... it is either continuous or ordinal.

        3. Binary variables (e.g., Mortality, Smoker, Event occurred): 
        - Variables that represent a YES/NO, TRUE/FALSE, or SUCCESS/FAILURE outcome
        - Reported as counts of people WITH the condition/outcome
        - Examples: "Number of deaths", "Smokers", "Complications occurred"
        - Usually reported as a count, total, or proportion of a binary outcome
        - If a variable is reported as a count or proportion of a binary outcome, use the binary category, not ordinal
        - If a variable is reported as a percentage or fraction of a group, treat it as binary

        4. Categorical variables (e.g., Race, Education level): 
        - Variables with distinct, named categories that are not binary outcomes
        - Examples: "Race: White/Black/Asian", "Education: High school/College/Graduate"
        - Categories represent different mutually exclusive types/classes, not counts of binary outcomes        

        RULE: When in doubt between categorical and binary, choose binary.

        If a binary variable cannot be intuitively mapped to 0/1 or True/False, specify the positive outcome in brackets (e.g., "Gender (Female)").

        IMPORTANT: Only extract variables from the REQUIRED VARIABLES list. Do not include any other measurements, even if they appear in tables.

        Return the data in the following JSON format:
        {{
            "variables": {{
                "continuous": [list of variable names from REQUIRED list],
                "ordinal": [list of variable names from REQUIRED list],
                "binary": [list of variable names from REQUIRED list],
                "categorical": [list of variable names from REQUIRED list]
            }}
        }}

        For variables not found in the paper, omit them from the output.
        Use the exact names from the REQUIRED VARIABLES list.
        """

        example_1 = {
            "input": """
        REQUIRED VARIABLES: Age, BMI, Gender (Female), Mortality

        | Variable | Treatment (n=50) | Control (n=48) |
        |----------|------------------|----------------|
        | Age, years | 65.2 ± 8.4 | 64.1 ± 9.2 |
        | BMI, kg/m² | 28.5 ± 4.1 | 27.9 ± 3.8 |
        | Gender, Female | 28 (56%) | 25 (52%) |
        | Mortality | 7 (14%) | 12 (25%) |
        | Height | 170 ± 8 | 168 ± 9 |
        | Blood pressure | 120/80 | 118/78 |
        """,
            "output": {
                "variables": {
                    "continuous": ["BMI"],
                    "ordinal": ["Age"],
                    "binary": ["Gender (Female)", "Mortality"],
                    "categorical": [],
                }
            },
        }

        example_2 = {
            "input": """
        REQUIRED VARIABLES: Wound infection, Readmission

        | Complication | Group A (n=60) | Group B (n=58) |
        |--------------|----------------|----------------|
        | Wound infection | 8 (13%) | 12 (21%) |
        | Bleeding | 5 (8%) | 7 (12%) |
        | Readmission | 12 (20%) | 15 (26%) |
        | Pain score | 3.2 ± 1.1 | 3.5 ± 1.2 |
        """,
            "output": {
                "variables": {
                    "continuous": [],
                    "ordinal": [],
                    "binary": ["Wound infection", "Readmission"],
                    "categorical": [],
                }
            },
        }

        examples = [example_1, example_2]

        existing = {
            "study_size": data.study_size,
            "groups": data.groups,
            "group_sizes": data.group_sizes,
        }

        result = self._llm_extract(
            content, system_prompt, existing, examples=examples
        )

        # Restructure the variables to match our model
        variables_dict = result.get("variables", {})
        variables_model = Variables()

        for var_type in ["continuous", "ordinal", "categorical", "binary"]:
            # Initialize empty structures for each identified variable
            for var_name in variables_dict.get(var_type, []):
                var_dict = {}
                for group in data.groups:
                    var_dict[group] = {}

                # Set to the corresponding type in our model
                getattr(variables_model, var_type)[var_name] = var_dict

        updated_data = ResearchData.model_construct(
            _fields_set={"variables"},
            study_size=data.study_size,
            groups=data.groups,
            group_sizes=data.group_sizes,
            variables=variables_model,
            statistical_tests=data.statistical_tests,
        )

        return updated_data

    def _extract_statistics(
        self, content: str, data: ResearchData
    ) -> ResearchData:
        """Extract statistical values for identified variables"""
        system_prompt = """
        You are a specialized statistical extractor focusing ONLY on extracting statistical values.

        CRITICAL SCOPE RESTRICTION:
        - The ONLY variables you are allowed to fill are those already present in the provided JSON under "variables".
        - DO NOT add new variables.
        - DO NOT add new groups.
        - DO NOT change variable names or move variables between types (continuous/ordinal/binary/categorical).
        - Ignore any measurements in the paper that are not already present in the provided JSON.

        TASK:
        For each variable already present in the JSON and for each existing group key, fill in the appropriate fields using evidence from the document.
        If a value is only reported for the total study, copy the SAME value into ALL groups.
        If you cannot find a value for a group, leave that group entry empty {{}}.

        
        1. For **continuous** variables: 
        - If normally distributed: 
            * Required: mean and standard deviation (std)
            * If available: skewness, minimum value (min_val), maximum value (max_val)
            * If heavy-tailed distribution is mentioned: tail_weight (0-1) and tail_std_factor

            Return in the following JSON format:
        {
            "variables": {
                "continuous": {
                    "variable_name": {
                        "group1": {
                            "mean": float_value,
                            "std": float_value,
                            "skew": optional_float_value,
                            "min_val": optional_float_value,
                            "max_val": optional_float_value,
                            "tail_weight": optional_float_value,
                            "tail_std_factor": optional_float_value
                        },
                        "group2": {...},
                        ...
                    }
                }
            }
        }
            

        - If not normally distributed: 
            * Required: median and either Q1 and Q3 derived from IQR as an interval, or IQR as a range 
            * Alternatively, ci95 (95% confidence interval) can be provided
            * If available: minimum value (min_val), maximum value (max_val)

        Return in the following JSON format:
        {
            "variables": {
                "continuous": {
                    "variable_name": {
                        "group1": {
                            "median": float_value,
                            "iqr": optional_list_of_float_values or string, # e.g., [Q1, Q3] or "Q1-Q3"
                            "q1": optional_float_value,
                            "q3": optional_float_value,
                            "ci95": optional_list_of_float_values or string, # e.g., [lower_bound, upper_bound] or "lower-upper"
                            "min_val": optional_float_value,
                            "max_val": optional_float_value
                        },
                        "group2": {...},
                        ...
                    }
                }
            }
        }
        
        2. For **ordinal** variables: 
        - Do not confuse with categorical variables, which have counts of distinct categories
        - If normally distributed: 
            * Required: mean and standard deviation (std)
            * If available: skewness, minimum value (min_val), maximum value (max_val)
            * If heavy-tailed distribution is mentioned: tail_weight (0-1) and tail_std_factor

            Return in the following JSON format:
        {
            "variables": {
                "ordinal": {
                    "variable_name": {
                        "group1": {
                            "mean": float_value,
                            "std": float_value,
                            "skew": optional_float_value,
                            "min_val": optional_float_value,
                            "max_val": optional_float_value,
                            "tail_weight": optional_float_value,
                            "tail_std_factor": optional_float_value
                        },
                        "group2": {...},
                        ...
                    }
                }
            }
        }
            

        - If not normally distributed: 
            * Required: median and either Q1 and Q3 derived from IQR as an interval, or IQR as a range 
            * Alternatively, ci95 (95% confidence interval) can be provided
            * If available: minimum value (min_val), maximum value (max_val)

        Return in the following JSON format:
        {
            "variables": {
                "ordinal": {
                    "variable_name": {
                        "group1": {
                            "median": float_value,
                            "iqr": optional_list_of_float_values or string, # e.g., [Q1, Q3] or "Q1-Q3"
                            "q1": optional_float_value,
                            "q3": optional_float_value,
                            "ci95": optional_list_of_float_values or string, # e.g., [lower_bound, upper_bound] or "lower-upper"
                            "min_val": optional_float_value,
                            "max_val": optional_float_value
                        },
                        "group2": {...},
                        ...
                    }
                }
            }
        }
        
        3. For **binary** variables:
        - For each group, extract BOTH:
            * The count of positive outcomes (numerator)
            * The total number of subjects for that variable in that group (denominator) (often indicated as "N" or "Total" in table headers)
            * Only extract the denominator if it is for that group specifically.
        - If the data is reported as "4/11", extract both 4 (count) and 11 (denominator).
        - If only a percentage is given (e.g., "36%"), use 36 as the count and 100 as the denominator.
        - Always specify which outcome is considered "positive" (e.g., "Mortality (Yes)").
        - If the denominator is not explicitly stated, use null.

    
        Return in the following JSON format:
        {
            "variables": {
                "binary": {
                    "variable_name (positive_outcome)": {
                        "group1": {
                            "count": int, 
                            "denominator": int or null},
                        "group2": {
                            "count": int, 
                            "denominator": int or null},
                        ...
                    }
                }
            }
        }
        
        4. For **categorical** variables:
        - Extract the count for each category in each group
        - Extract the total count for each group (often indicated as "N" or "Total" in table headers)
        - Make sure to extract ALL categories mentioned

        Return in the following JSON format:
        {
            "variables": {
                "categorical": {
                    "variable_name": {
                        "group1": {
                            "category1": int_count,
                            "category2": int_count,
                            ...
                            "total": int_count
                        },
                        "group2": {...},
                        ...
                    }
                }
            }
        }
        
        For distributions described as "skewed" or having "heavy tails", extract skewness parameters
        when possible. For variables with explicit value ranges (e.g., "BMI ranged from 18.5 to 42.3"),
        extract min_val and max_val.
        
        Return in valid JSON format matching the variable structure, with each variable containing
        the appropriate statistical information for each group. DO NOT use any other structure than the one for its dedicated variable type.
        """

        example_1 = {
            "input": """
        | Variable | Total Study (N=120) |
        |----------|---------------------|
        | BMI, kg/m² | 28.2 ± 4.0 |
        | Age, years | 65.5 ± 8.2 |

        Overall study statistics show mean BMI was 28.2 kg/m² with standard deviation of 4.0.
        Mean age across all participants was 65.5 years (SD = 8.2).

        Previously extracted variables: {"continuous": {"BMI": {}}, "ordinal": {"Age": {}}}
        Previously extracted groups: ["Treatment A", "Treatment B", "Control"]
        """,
            "output": {
                "variables": {
                    "continuous": {
                        "BMI": {
                            "Treatment A": {"mean": 28.2, "std": 4.0},
                            "Treatment B": {"mean": 28.2, "std": 4.0},
                            "Control": {"mean": 28.2, "std": 4.0},
                        }
                    },
                    "ordinal": {
                        "Age": {
                            "Treatment A": {"mean": 65.5, "std": 8.2},
                            "Treatment B": {"mean": 65.5, "std": 8.2},
                            "Control": {"mean": 65.5, "std": 8.2},
                        }
                    },
                }
            },
        }

        example_2 = {
            "input": """
        | Variable | Treatment (n=50) | Control (n=48) | Placebo (n=45) |
        |----------|------------------|----------------|----------------|
        | BMI, kg/m² | 28.5 ± 4.1 | 27.9 ± 3.8 | - |
        | Mortality | 7 (14%) | 12 (25%) | Not reported |
        | Complications | 15/50 | - | 8/45 |

        Previously extracted variables: {
            "continuous": {"BMI": {}}, 
            "binary": {"Mortality": {}, "Complications": {}}
        }
        Previously extracted groups: ["Treatment", "Control", "Placebo"]
        """,
            "output": {
                "variables": {
                    "continuous": {
                        "BMI": {
                            "Treatment": {"mean": 28.5, "std": 4.1},
                            "Control": {"mean": 27.9, "std": 3.8},
                            "Placebo": {},
                        }
                    },
                    "binary": {
                        "Mortality": {
                            "Treatment": {"count": 7, "denominator": 50},
                            "Control": {"count": 12, "denominator": 48},
                            "Placebo": {},
                        },
                        "Complications": {
                            "Treatment": {"count": 15, "denominator": 50},
                            "Control": {},
                            "Placebo": {"count": 8, "denominator": 45},
                        },
                    },
                }
            },
        }

        example_3 = {
            "input": """
        Overall Education Distribution (N=150):
        - High school: 60 participants (40%)
        - College: 75 participants (50%) 
        - Graduate: 15 participants (10%)

        Previously extracted variables: {"categorical": {"Education": {}}}
        Previously extracted groups: ["Group A", "Group B"]
        Previously extracted group_sizes: {"Group A": 75, "Group B": 75}
        """,
            "output": {
                "variables": {
                    "categorical": {
                        "Education": {
                            "Group A": {
                                "High school": 60,
                                "College": 75,
                                "Graduate": 15,
                                "total": 150,
                            },
                            "Group B": {
                                "High school": 60,
                                "College": 75,
                                "Graduate": 15,
                                "total": 150,
                            },
                        }
                    }
                }
            },
        }

        examples = [example_1, example_2, example_3]
        existing = data.model_dump()
        result = self._llm_extract(
            content, system_prompt, existing, examples=examples
        )

        # Extract variables from result
        result_vars = result.get("variables", {})

        # Update our existing variables model with the statistics
        updated_variables = data.variables.model_copy(deep=True)

        # For each variable type, update with the extracted statistics
        for var_type in ["continuous", "ordinal", "categorical", "binary"]:
            type_vars = result_vars.get(var_type, {})
            for var_name, var_stats in type_vars.items():
                if var_name in getattr(updated_variables, var_type):
                    getattr(updated_variables, var_type)[var_name] = var_stats

        # Construct updated data
        updated_data = ResearchData.model_construct(
            _fields_set={"variables"},
            study_size=data.study_size,
            groups=data.groups,
            group_sizes=data.group_sizes,
            variables=updated_variables,
            statistical_tests=data.statistical_tests,
        )

        return updated_data

    def _refine_variables(
        self, content: str, data: ResearchData
    ) -> ResearchData:

        system_prompt = """
        You are a specialized statistical extractor. STRICTLY follow these rules:

        1. DATA TYPE CORRECTION: 
           If a variable's data does not fit the template for its assigned type, MOVE it to the correct type.
           - Example: If "Age" is in continuous but has median/IQR, move it to ordinal.
           - Example: If "Gender" is in binary but has 3+ categories (Male/Female/Other), move it to categorical.

        2. PRESERVE VARIABLE NAMES: 
           Do NOT rename, merge, or split variables. Keep the exact names provided in the input.
           Your goal is ONLY to fix structural/type errors, not to reorganize the data.

        3. STRUCTURE ENFORCEMENT:
           For each variable, use ONLY the structure shown in the templates below for its type. 
           Use null for missing or unknown values. Do not invent or guess data.

        EXAMPLES (use exactly this structure, but with real variable and group names):

        Continuous: numerical variables with a continuous range of possible values, such as weight, height, or BMI.
        {
        "variables": {
            "continuous": {
            "BMI": {
                "group_1": {
                    "mean": 24.5,
                    "std": 2.1,
                    "min_val": 18.5,
                    "max_val": 42.3
                },
                "group_2": {
                    "median": 25.1,
                    "iqr": [22.0, 28.0],
                    "min_val": 19.0,
                    "max_val": 40.0
                }
            }
            }
        }
        }

        Ordinal: numeric variables that describe a distribution of discrete values (without decimal points) in their range of possible values, such as Age, Score, or Rating.
        {
        "variables": {
            "ordinal": {
            "Age": {
                "group_1": {
                    "mean": 24.5,
                    "std": 2.1,
                    "min_val": 18.5,
                    "max_val": 42.3
                },
                "group_2": {
                    "median": 25.1,
                    "iqr": [22.0, 28.0],
                    "min_val": 19.0,
                    "max_val": 40.0
                }
            }
            }
        }
        }

        Categorical: variables with distinct, mutually exclusive named categories that are not binary outcomes, such as race, education level, or blood type.
        {
        "variables": {
            "categorical": {
            "Race": {
                "group_1": {
                    "White": 20,
                    "Black": 10,
                    "total": 30
                },
                "group_2": {
                    "White": 18,
                    "Black": 12,
                    "total": 30
                }
            }
            }
        }
        }

        Binary: variables that represent a YES/NO, TRUE/FALSE, or SUCCESS/FAILURE outcome, such as Mortality, Smoker, or Event occurred.
        {
        "variables": {
            "binary": {
            "Mortality": {
                "group1": {
                    "count": 7,
                    "denominator": 18
                },
                "group2": {
                    "count": 5,
                    "denominator": 22
                }
            }
            }
        }
        }

        4. VALIDATION:
           - Every variable must match the template for its type.
           - Group names must match exactly.
           - Denominators for binary variables must not exceed group size.

        5. Return ONLY valid JSON matching these examples.
        """

        example_1 = {
            "input": """
        Current variables with mutually exclusive marital status:
        {
            "binary": {
                "Married": {"Group A": {"count": 25, "denominator": 50}},
                "Single": {"Group A": {"count": 15, "denominator": 50}},
                "Divorced": {"Group A": {"count": 10, "denominator": 50}}
            }
        }
        """,
            "output": {
                "variables": {
                    "binary": {},
                    "categorical": {
                        "Marital Status": {
                            "Group A": {
                                "Married": 25,
                                "Single": 15,
                                "Divorced": 10,
                                "total": 50,
                            }
                        }
                    },
                }
            },
        }
        example_2 = {
            "input": """
        Current variables with Age incorrectly in continuous:
        {
            "continuous": {
                "Age": {"Group A": {"mean": 65.2, "std": 8.4}}
            }
        }
        """,
            "output": {
                "variables": {
                    "continuous": {},
                    "ordinal": {
                        "Age": {"Group A": {"mean": 65.2, "std": 8.4}}
                    },
                }
            },
        }

        example_3 = {
            "input": """
        Current variables with mutually exclusive race categories:
        {
            "binary": {
                "White": {"Treatment": {"count": 32, "denominator": 60}, "Control": {"count": 28, "denominator": 55}},
                "Black": {"Treatment": {"count": 18, "denominator": 60}, "Control": {"count": 17, "denominator": 55}},
                "Asian": {"Treatment": {"count": 7, "denominator": 60}, "Control": {"count": 8, "denominator": 55}},
                "Other": {"Treatment": {"count": 3, "denominator": 60}, "Control": {"count": 2, "denominator": 55}}
            }
        }
        """,
            "output": {
                "variables": {
                    "binary": {},
                    "categorical": {
                        "Race": {
                            "Treatment": {
                                "White": 32,
                                "Black": 18,
                                "Asian": 7,
                                "Other": 3,
                                "total": 60,
                            },
                            "Control": {
                                "White": 28,
                                "Black": 17,
                                "Asian": 8,
                                "Other": 2,
                                "total": 55,
                            },
                        }
                    },
                }
            },
        }

        examples = [example_1, example_2, example_3]
        existing = data.model_dump()
        result = self._llm_extract(
            content, system_prompt, existing, examples=examples
        )

        # Extract variables from result
        result_vars = result.get("variables", {})

        updated_variables = Variables()

        for var_type in ["continuous", "ordinal", "categorical", "binary"]:
            type_vars = result_vars.get(var_type, {})
            for var_name, var_stats in type_vars.items():
                # Set the variable data directly (don't check if it exists)
                getattr(updated_variables, var_type)[var_name] = var_stats

        updated_data = ResearchData.model_construct(
            _fields_set={"variables"},
            study_size=data.study_size,
            groups=data.groups,
            group_sizes=data.group_sizes,
            variables=updated_variables,
            statistical_tests=data.statistical_tests,
        )

        return updated_data

    def _extract_tests(self, content: str, data: ResearchData) -> ResearchData:
        """Extract statistical tests and relationships between variables"""

        # Build DAG guidance if available
        dag_guidance = ""
        target_edges = []
        if self.dag is not None:
            # Format edges as "VarA - VarB"
            for u, v in self.dag.edges():
                target_edges.append(f"{u} - {v}")

            dag_guidance = f"""
        CRITICAL FOCUS - DAG RELATIONSHIPS:
        We are specifically interested in statistical tests that validate the following relationships:
        {json.dumps(target_edges, indent=2)}
        
        Please prioritized finding tests that correspond to these pairs. 
        For each pair (A - B), look for:
        - Correlation tests between A and B
        - T-tests/ANOVA/Mann-Whitney where A is the grouping variable and B is the outcome
        - Chi-square tests between A and B
        - Regression coefficients where A predicts B
        
        You may also extract other significant tests mentioned in the text, but the DAG pairs are the priority.
        """

        system_prompt = f"""
        You are a specialized statistical extractor focusing ONLY on extracting statistical tests.
        {dag_guidance}
        
        For each statistical test mentioned in the document, extract:
        1. The variables involved (names must match previously identified variables)
        2. The type of test - MUST be one of the following allowed values:
        - "pearson" (for Pearson correlation coefficient)
        - "spearman" (for Spearman rank correlation)
        - "chi_square" (for chi-squared tests of independence or goodness of fit)
        - "mcnemar" (for McNemar's test for paired nominal data)
        - "wilcoxon_signed_rank" (for Wilcoxon signed-rank test)
        - "paired_t_test" (for paired/dependent samples t-test)
        - "wilcoxon_mann_whitney" (for Wilcoxon-Mann-Whitney U test)
        - "unpaired_t_test" (for independent/unpaired samples t-test)
        - "friedman" (for Friedman test)
        - "kruskal_wallis" (for Kruskal-Wallis H test)
        - "one_way_anova" (for one-way analysis of variance)
        - "ranova" (for repeated measures ANOVA)
        3. The test statistic value (this is the result of the test, e.g., t-value, F-value, etc.)
        4. The p-value
        5. Any effect size mentioned (this is a measure of the strength of the relationship, e.g., Cohen's d, eta-squared, prevalence ratio, etc.)
        6. The groups involved (only use groups previously identified in the study)


        If a test is conducted for one variable across multiple groups, mention only one variable and the groups.
        
        IMPORTANT: The test_type field MUST be exactly one of the values listed above. For example:
        - If you see "t-test for independent samples", use "unpaired_t_test"
        - If you see "Student's t-test between groups", use "unpaired_t_test"
        - If you see "Pearson correlation coefficient", use "pearson"
        - If you see "ANOVA", use "one_way_anova"


        IMPORTANT: If there is none of the above tests mentioned, do not assume or create any tests.
        
        Return in the following JSON format:
        {{
            "statistical_tests": [
                {{
                    "variables": ["variable1", "variable2"],
                    "test_type": "valid_test_type_from_list_above",
                    "test_statistic": float_value,
                    "p_value": float_value,
                    "effect_size": optional_float_value,
                    "group_means": optional_list_of_means,
                    "groups": ["group1", "group2"]
                }},
                ...
            ]
        }}


        CRITICAL: Use EXACT group names from the previously extracted groups. 
        Previously extracted groups: {data.groups}
        If you encounter group names in the text that don't exactly match these, map them as follows:
        - Look for partial matches (e.g., "control group" → "Control")
        - Use the closest matching previously extracted group name
        - If no reasonable match exists, use "Unknown Group"


        GROUP VALIDATION: Every test must use only groups from the above list.
        """

        example_1 = {
            "input": """ Baseline characteristics were compared using independent t-tests for continuous  variables and chi-square tests for categorical variables. The mean age was  significantly different between groups (t=2.45, p=0.016). Gender distribution  did not differ significantly between groups (χ²=1.23, p=0.267).   Previously extracted groups: ["Treatment", "Control"] Previously extracted variables: {"ordinal": ["Age"], "binary": ["Gender (Female)"]} """,
            "output": {
                "statistical_tests": [
                    {
                        "variables": ["Age"],
                        "test_type": "unpaired_t_test",
                        "test_statistic": 2.45,
                        "p_value": 0.016,
                        "effect_size": None,
                        "group_means": None,
                        "groups": ["Treatment", "Control"],
                    },
                    {
                        "variables": ["Gender (Female)"],
                        "test_type": "chi_square",
                        "test_statistic": 1.23,
                        "p_value": 0.267,
                        "effect_size": None,
                        "group_means": None,
                        "groups": ["Treatment", "Control"],
                    },
                ]
            },
        }

        example_2 = {
            "input": """ One-way ANOVA was used to compare BMI across the three treatment groups  (F=4.52, p=0.012). Post-hoc analysis revealed significant differences.  Mean BMI was 24.8 kg/m² in the Low Dose group, 26.3 kg/m² in the High Dose group,  and 28.1 kg/m² in the Control group. Pearson correlation showed a strong positive relationship between BMI and  weight (r=0.78, p<0.001).   Previously extracted groups: ["Low Dose", "High Dose", "Control"] Previously extracted variables: {"continuous": ["BMI", "Weight"]} """,
            "output": {
                "statistical_tests": [
                    {
                        "variables": ["BMI"],
                        "test_type": "one_way_anova",
                        "test_statistic": 4.52,
                        "p_value": 0.012,
                        "effect_size": None,
                        "group_means": [24.8, 26.3, 28.1],
                        "groups": ["Low Dose", "High Dose", "Control"],
                    },
                    {
                        "variables": ["BMI", "Weight"],
                        "test_type": "pearson",
                        "test_statistic": 0.78,
                        "p_value": 0.001,
                        "effect_size": None,
                        "group_means": None,
                        "groups": ["Low Dose", "High Dose", "Control"],
                    },
                ]
            },
        }

        examples = [example_1, example_2]

        existing = {
            "study_size": data.study_size,
            "groups": data.groups,
            "group_sizes": data.group_sizes,
            "variables": data.variables.model_dump(),  # Pass full variable list so LLM knows valid names
        }

        result = self._llm_extract(
            content, system_prompt, existing, examples=examples
        )

        # Extract and validate tests
        tests = []
        for test_data in result.get("statistical_tests", []):
            try:

                test = StatisticalTest.model_validate(test_data)
                tests.append(test)
            except Exception as e:
                print(f"Error validating test: {e}")

        # Construct updated data
        updated_data = ResearchData.model_construct(
            _fields_set={"statistical_tests"},
            study_size=data.study_size,
            groups=data.groups,
            group_sizes=data.group_sizes,
            variables=data.variables,
            statistical_tests=tests,
        )

        return updated_data
