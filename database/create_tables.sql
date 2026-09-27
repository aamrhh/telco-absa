

CREATE TABLE raw_reviews (
    review_id INT AUTO_INCREMENT PRIMARY KEY,
    telco VARCHAR(50) NOT NULL,
    review_text TEXT NOT NULL,
    rating INT NULL,
    review_date DATE NULL,
    source VARCHAR(50) NOT NULL
);

CREATE TABLE processed_reviews (
    processed_id INT AUTO_INCREMENT PRIMARY KEY,
    review_id INT,
    telco VARCHAR(50) NOT NULL,
    language_version VARCHAR(20) NOT NULL,
    processed_text TEXT NOT NULL,
    FOREIGN KEY (review_id) REFERENCES raw_reviews(review_id)
);

CREATE TABLE aspect_sentiment (
    sentiment_id INT AUTO_INCREMENT PRIMARY KEY,
    processed_id INT,
    telco VARCHAR(50) NOT NULL,
    aspect VARCHAR(50) NULL,
    language_version VARCHAR(20) NOT NULL,
    manual_label VARCHAR(20) NOT NULL,
    predicted_sentiment VARCHAR(20) NOT NULL,
    is_single_aspect BOOLEAN NOT NULL,
    FOREIGN KEY (processed_id) REFERENCES processed_reviews(processed_id)
);

CREATE TABLE analyzer_results (
    analyzer_id INT AUTO_INCREMENT PRIMARY KEY,
    input_text TEXT NOT NULL,
    detected_aspect VARCHAR(50) NOT NULL,
    predicted_sentiment VARCHAR(20) NOT NULL,
    timestamp DATETIME NOT NULL
);