"""Unit tests for resume data models."""

from datetime import date

import pytest
from pydantic import ValidationError

from myjobspyai.models.resume import (
    Education,
    EducationLevel,
    Experience,
    ExperienceLevel,
    ResumeData,
    Skill,
    SkillCategory,
)


def test_education_model():
    """Test the Education model creation and validation."""
    # Test valid education entry
    education = Education(
        institution="Test University",
        degree="Bachelor of Science",
        field_of_study="Computer Science",
        level=EducationLevel.BACHELORS,
        start_date=date(2018, 9, 1),
        end_date=date(2022, 5, 15),
        gpa=3.7,
        description="Studied computer science fundamentals",
    )

    assert education.institution == "Test University"
    assert education.degree == "Bachelor of Science"
    assert education.field_of_study == "Computer Science"
    assert education.level == EducationLevel.BACHELORS
    assert education.gpa == 3.7
    assert "fundamentals" in education.description

    # Test date validation (end date before start date)
    with pytest.raises(ValueError, match="End date must be after start date"):
        Education(
            institution="Test University",
            degree="Bachelor of Science",
            field_of_study="Computer Science",
            level=EducationLevel.BACHELORS,
            start_date=date(2022, 9, 1),
            end_date=date(2020, 5, 15),
        )

    # Test GPA validation
    with pytest.raises(ValidationError):
        Education(
            institution="Test University",
            degree="Bachelor of Science",
            field_of_study="Computer Science",
            level=EducationLevel.BACHELORS,
            gpa=5.0,  # Invalid GPA
        )


def test_experience_model():
    """Test the Experience model creation and validation."""
    # Test valid experience entry
    experience = Experience(
        company="Tech Corp",
        position="Software Engineer",
        location="San Francisco, CA",
        current=True,
        description="Developed web applications using Python and React",
        skills_used=["Python", "React", "Django"],
        achievements=["Improved performance by 30%"],
        experience_level=ExperienceLevel.MID,
    )

    assert experience.company == "Tech Corp"
    assert experience.position == "Software Engineer"
    assert "Python" in experience.skills_used
    assert "Improved" in experience.achievements[0]
    assert experience.experience_level == ExperienceLevel.MID

    # Test date validation
    with pytest.raises(ValueError, match="End date must be after start date"):
        Experience(
            company="Tech Corp",
            position="Software Engineer",
            start_date=date(2023, 1, 1),
            end_date=date(2022, 1, 1),
            description="Test",
        )


def test_skill_model():
    """Test the Skill model creation and validation."""
    # Test valid skill
    skill = Skill(
        name="Python",
        category=SkillCategory.PROGRAMMING,
        proficiency=0.9,
        years_experience=5,
        last_used=2023,
    )

    assert skill.name == "Python"
    assert skill.category == SkillCategory.PROGRAMMING
    assert skill.proficiency == 0.9
    assert skill.years_experience == 5
    assert skill.last_used == 2023

    # Test proficiency validation
    with pytest.raises(ValidationError):
        Skill(
            name="Python",
            category=SkillCategory.PROGRAMMING,
            proficiency=1.1,  # Invalid proficiency
        )


def test_resume_data_model():
    """Test the ResumeData model creation and validation."""
    # Create test data
    education = Education(
        institution="Test University",
        degree="Bachelor of Science",
        field_of_study="Computer Science",
        level=EducationLevel.BACHELORS,
    )

    experience = Experience(
        company="Tech Corp",
        position="Software Engineer",
        description="Developed applications",
    )

    skill = Skill(name="Python", category=SkillCategory.PROGRAMMING)

    # Test valid resume
    resume = ResumeData(
        full_name="John Doe",
        email="john.doe@example.com",
        education=[education],
        experience=[experience],
        skills=[skill],
        certifications=[{"name": "AWS Certified", "issuer": "Amazon"}],
        languages=[{"name": "English", "proficiency": "native"}],
        projects=[{"name": "Project X", "description": "A test project"}],
    )

    assert resume.full_name == "John Doe"
    assert len(resume.education) == 1
    assert len(resume.experience) == 1
    assert len(resume.skills) == 1

    # Test email validation
    with pytest.raises(ValueError, match="Invalid email format"):
        ResumeData(
            email="invalid-email",
            education=[education],
            experience=[experience],
            skills=[skill],
        )

    # Test skills validation (at least one skill required)
    with pytest.raises(ValueError, match="At least one skill is required"):
        ResumeData(
            email="test@example.com",
            education=[education],
            experience=[experience],
            skills=[],
        )


def test_serialization_deserialization():
    """Test serialization and deserialization of models."""
    # Create test data
    skill = Skill(name="Python", category=SkillCategory.PROGRAMMING, proficiency=0.9)

    education = Education(
        institution="Test University",
        degree="Bachelor of Science",
        field_of_study="Computer Science",
        level=EducationLevel.BACHELORS,
        start_date=date(2018, 9, 1),
    )

    experience = Experience(
        company="Tech Corp",
        position="Software Engineer",
        start_date=date(2020, 1, 1),
        description="Developer",
    )

    # Create and serialize resume
    resume = ResumeData(
        full_name="John Doe",
        email="john@example.com",
        education=[education],
        experience=[experience],
        skills=[skill],
    )

    # Convert to dict and back
    resume_dict = resume.model_dump()
    deserialized = ResumeData.model_validate(resume_dict)

    # Verify data integrity
    assert deserialized.full_name == resume.full_name
    assert len(deserialized.education) == 1
    assert len(deserialized.experience) == 1
    assert len(deserialized.skills) == 1
    assert deserialized.education[0].institution == "Test University"
    assert deserialized.experience[0].company == "Tech Corp"
    assert deserialized.skills[0].name == "Python"


def test_resume_data_helper_methods():
    """Test the helper methods in ResumeData."""
    # Create test data
    python_skill = Skill(
        name="Python", category=SkillCategory.PROGRAMMING, proficiency=0.9
    )

    aws_skill = Skill(name="AWS", category=SkillCategory.CLOUD, proficiency=0.8)

    education = Education(
        institution="Test University",
        degree="Bachelor of Science",
        field_of_study="Computer Science",
        level=EducationLevel.BACHELORS,
        start_date=date(2018, 9, 1),
        end_date=date(2022, 5, 15),
    )

    experience = Experience(
        company="Tech Corp",
        position="Software Engineer",
        start_date=date(2022, 1, 1),
        end_date=date(2023, 1, 1),
        description="Developer",
    )

    # Create resume
    resume = ResumeData(
        full_name="John Doe",
        email="john@example.com",
        education=[education],
        experience=[experience],
        skills=[python_skill, aws_skill],
    )

    # Test get_skills_by_category
    programming_skills = resume.get_skills_by_category(SkillCategory.PROGRAMMING)
    assert len(programming_skills) == 1
    assert programming_skills[0].name == "Python"

    # Test get_highest_education
    highest_ed = resume.get_highest_education()
    assert highest_ed is not None
    assert highest_ed.level == EducationLevel.BACHELORS

    # Test get_total_experience_years
    total_exp = resume.get_total_experience_years()
    assert 0.9 <= total_exp <= 1.1  # Approximately 1 year of experience
