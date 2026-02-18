import { Carousel } from 'react-bootstrap';
import './MyCarousel.css';

const MyCarousel = ({
  images,
  alts,
}) => {
  const checkImages = images || [];
  const checkAlts = alts || [];

  return (<>
    <Carousel interval={7000} controls={true} indicators={true}>
      {checkImages.map((element, idx) => (
      <Carousel.Item key={`img${idx}`}>
        <img
          className="d-block w-100"
          src={element || null}
          alt={checkAlts[idx] || null}
        />
      </Carousel.Item>
      ))}
    </Carousel>
  </>)
};

export default MyCarousel;
