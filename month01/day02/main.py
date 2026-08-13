import logging


from validator import calculate_max, calculate_average



def main():
    logging.basicConfig(level=logging.INFO)
    
    numbers = []

    try:
        maximum = calculate_max(numbers)
        average = calculate_average(numbers)

        logging.info(f"Maximum: {maximum}")
        logging.info(f"Average: {average}")

    except ValueError as error:
        logging.error(f"Error: {error}")


if __name__ == "__main__":
    main()